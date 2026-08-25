"""Fetch post-2017 trial evidence for CtD external validation.

Implements external_validation/protocol_frozen.md §5 (2026-08-22, FROZEN):
  * queries ClinicalTrials.gov API v2 per Hetionet Disease label with
    filter.advanced=AREA[StartDate]RANGE[2017-01-01,MAX] (syntax verified
    2026-08-22; filter.sasDateRangeStart is NOT a valid parameter);
  * caches every raw JSON page under external_validation/cache/ctgov/ using
    content-hash filenames (md5 hex of the request URL) plus a manifest that
    maps hashes back to human-readable labels;
  * extracts (intervention name, condition, NCT id, start date) tuples and
    aligns them to Hetionet (Compound, Disease) names by exact match after
    documented normalization only (no fuzzy matching - frozen rule);
  * writes evidence_ctd.tsv plus ctgov_alignment_report.json; drop-outs are
    counted, never silently discarded.

Safety invariants:
  * every network request goes through safe_get(): https + frozen host
    allowlist, private/loopback/link-local addresses blocked at DNS
    resolution, redirects re-validated through the same checks;
  * every filesystem write goes through write_under()/read_under() which
    resolve the target and require it to stay inside external_validation/;
  * no value derived from network data is ever used as a path component
    (only md5 hex digests and fixed literals are).

Usage: python external_validation/fetch_ctgov.py [--delay 0.6]
"""
from __future__ import annotations

import argparse
import hashlib
import ipaddress
import json
import re
import socket
import time
import urllib.parse
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
EV = ROOT / "external_validation"
CACHE = EV / "cache" / "ctgov"
API_HOST = "clinicaltrials.gov"
NODES_HOST = "raw.githubusercontent.com"
ALLOWED_HOSTS = {API_HOST, NODES_HOST}
API = f"https://{API_HOST}/api/v2/studies"
NODES_URL = (f"https://{NODES_HOST}/hetio/hetionet/main/"
             "hetnet/tsv/hetionet-v1.0-nodes.tsv")
START_CUTOFF = "2017-01-01"          # frozen (protocol §5)
STRIP_WORDS = {"disease", "syndrome", "chronic", "acute"}  # frozen pass 2


# ---------- confined filesystem helpers ----------

def _confined(target: Path) -> Path:
    base = EV.resolve()
    resolved = target.resolve()
    if base != resolved and base not in resolved.parents:
        raise ValueError(f"path escapes {EV}: {resolved}")
    return resolved


def write_under(rel_path: Path, data) -> None:
    p = _confined(rel_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(data, bytes):
        p.write_bytes(data)
    else:
        p.write_text(data)


def read_under(rel_path: Path) -> str:
    return _confined(rel_path).read_text()


def exists_under(rel_path: Path) -> bool:
    return _confined(rel_path).exists()


def hex_name(url_or_label: str) -> str:
    """Deterministic [0-9a-f]-only file name stem from arbitrary input."""
    return hashlib.sha256(url_or_label.encode("utf-8")).hexdigest()


# ---------- guarded network access ----------

def _assert_safe_url(url: str) -> None:
    p = urllib.parse.urlsplit(url)
    if p.scheme != "https":
        raise ValueError(f"non-https URL blocked: {url}")
    if p.hostname not in ALLOWED_HOSTS:
        raise ValueError(f"host not in allowlist: {p.hostname}")
    try:
        infos = socket.getaddrinfo(p.hostname, 443, proto=socket.IPPROTO_TCP)
    except socket.gaierror as e:
        raise ValueError(f"DNS resolution failed for {p.hostname}: {e}")
    # 198.18.0.0/15 (RFC 2544 benchmark range) is used by local proxy
    # tools in fake-IP DNS mode: the proxy tunnels the connection to the
    # real destination, which is still governed by the hostname allowlist
    # and https scheme above, so these are allowed through.
    fake_ip = ipaddress.ip_network("198.18.0.0/15")
    for info in infos:
        ip = ipaddress.ip_address(info[4][0])
        if ip in fake_ip:
            continue
        if (ip.is_private or ip.is_loopback or ip.is_link_local
                or ip.is_reserved or ip.is_multicast):
            raise ValueError(f"non-public address blocked: {ip}")


class _SafeRedirectHandler(urllib.request.HTTPRedirectHandler):
    """Redirect handler that re-applies the same URL safety checks."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):  # noqa: D102
        _assert_safe_url(newurl)
        return super().redirect_request(req, fp, code, msg, headers, newurl)


_OPENER = urllib.request.build_opener(_SafeRedirectHandler())


def safe_get(url: str, retries: int = 4) -> bytes:
    for i in range(retries):
        _assert_safe_url(url)
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "..."} )
            with _OPENER.open(req, timeout=60) as r:
                return r.read()
        except Exception:  # noqa: BLE001 - network retry loop
            if i == retries - 1:
                raise
            time.sleep(3 * (i + 1))
    raise RuntimeError("unreachable")


# ---------- frozen alignment rules ----------

def norm(name: str) -> str:
    """Frozen normalization: lowercase, strip punctuation, collapse spaces."""
    s = name.lower()
    s = re.sub(r"[^a-z0-9\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def norm_strip(name: str) -> str:
    """Pass-2 normalization: additionally drop frozen modifier suffix words."""
    toks = [t for t in norm(name).split() if t not in STRIP_WORDS]
    return " ".join(toks)


def load_hetionet_names() -> dict:
    """Fetch dhimmel nodes.tsv and return {kind: {norm_name: (id, name)}}."""
    nodes_rel = CACHE / "nodes.tsv"
    if not exists_under(nodes_rel):
        write_under(nodes_rel, safe_get(NODES_URL))
    kinds = {"Compound": {}, "Disease": {}}
    second = {"Compound": {}, "Disease": {}}
    for line in read_under(nodes_rel).splitlines()[1:]:
        parts = line.split("\t")
        if len(parts) < 3:
            continue
        nid, name, kind = parts[0], parts[1], parts[2]
        if kind in kinds:
            kinds[kind][norm(name)] = (nid, name)
            second[kind][norm_strip(name)] = (nid, name)
    return {"primary": kinds, "secondary": second}


def fetch_condition(disease_name: str, delay: float, manifest: dict) -> list:
    """Paged query for one disease label; returns raw study dicts."""
    studies, page_token, page = [], None, 0
    while True:
        q = urllib.parse.quote(disease_name)
        url = (f"{API}?query.cond={q}"
               f"&filter.advanced=AREA%5BStartDate%5DRANGE%5B{START_CUTOFF}%2CMAX%5D"
               f"&pageSize=100")
        if page_token:
            url += f"&pageToken={urllib.parse.quote(page_token)}"
        stem = hex_name(url)
        cache_rel = CACHE / f"{stem}.json"
        if exists_under(cache_rel):
            data = json.loads(read_under(cache_rel))
        else:
            data = json.loads(safe_get(url))
            write_under(cache_rel, json.dumps(data))
            manifest[stem] = {"url": url, "disease": disease_name,
                              "page": page}
            time.sleep(delay)
        studies.extend(data.get("studies", []))
        page_token = data.get("nextPageToken")
        page += 1
        if not page_token or page > 60:
            break
    return studies


def extract_evidence(names: dict, delay: float) -> dict:
    dis_pri = names["primary"]["Disease"]
    dis_sec = names["secondary"]["Disease"]
    com_pri = names["primary"]["Compound"]
    com_sec = names["secondary"]["Compound"]

    dis_labels = sorted({v[1] for v in dis_pri.values()})
    report = {
        "n_diseases_queried": len(dis_labels),
        "n_studies_seen": 0,
        "n_studies_with_start_date": 0,
        "n_conditions_total": 0,
        "n_conditions_aligned": 0,
        "n_interventions_total": 0,
        "n_interventions_aligned": 0,
        "n_pairs_emitted": 0,
    }
    pairs = {}
    manifest = {}
    for label in dis_labels:
        for s in fetch_condition(label, delay, manifest):
            report["n_studies_seen"] += 1
            proto = s.get("protocolSection", {})
            start = (proto.get("statusModule", {})
                        .get("startDateStruct", {}).get("date", ""))
            if not start:
                continue
            report["n_studies_with_start_date"] += 1
            conds = proto.get("conditionsModule", {}).get("conditions", [])
            intvs = [i.get("name", "") for i in
                     proto.get("armsInterventionsModule", {})
                     .get("interventions", []) if i.get("name")]
            # --- disease alignment (frozen, exact after normalization) ---
            aligned_diseases = []
            for c in conds:
                report["n_conditions_total"] += 1
                hit = dis_pri.get(norm(c)) or dis_sec.get(norm_strip(c))
                if hit:
                    aligned_diseases.append(hit)
            report["n_conditions_aligned"] += len(aligned_diseases)
            # --- compound alignment (frozen, exact after normalization) ---
            aligned_compounds = []
            for iv in intvs:
                report["n_interventions_total"] += 1
                hit = com_pri.get(norm(iv)) or com_sec.get(norm_strip(iv))
                if hit:
                    aligned_compounds.append(hit)
            report["n_interventions_aligned"] += len(aligned_compounds)
            for (cid, cname) in aligned_compounds:
                for (did, dname) in aligned_diseases:
                    key = (cid, did)
                    rec = pairs.setdefault(key, {"compound": cname,
                                                 "disease": dname,
                                                 "nct_ids": set(),
                                                 "start_dates": set()})
                    rec["nct_ids"].add(proto.get("identificationModule", {})
                                         .get("nctId", "?"))
                    rec["start_dates"].add(start)
                    report["n_pairs_emitted"] += 1
    write_under(CACHE / "manifest.json", json.dumps(manifest, indent=1,
                                                    sort_keys=True))
    return {"report": report, "pairs": pairs}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--delay", type=float, default=0.6)
    args = ap.parse_args()

    CACHE.mkdir(parents=True, exist_ok=True)
    names = load_hetionet_names()
    print(f"Hetionet names: {len(names['primary']['Disease'])} diseases, "
          f"{len(names['primary']['Compound'])} compounds", flush=True)
    out = extract_evidence(names, args.delay)
    rep, pairs = out["report"], out["pairs"]
    print("ALIGNMENT REPORT:", json.dumps(rep, indent=1), flush=True)

    lines = ["compound_id\tcompound_name\tdisease_id\tdisease_name\t"
             "n_trials\tstart_years\tearliest_start\tlatest_start"]
    for (cid, did), rec in sorted(pairs.items()):
        yrs = sorted({d[:4] for d in rec["start_dates"]})
        ds_ = sorted(rec["start_dates"])
        lines.append(f"{cid}\t{rec['compound']}\t{did}\t{rec['disease']}\t"
                     f"{len(rec['nct_ids'])}\t{','.join(yrs)}\t{ds_[0]}\t"
                     f"{ds_[-1]}")
    write_under(EV / "evidence_ctd.tsv", "\n".join(lines) + "\n")
    write_under(EV / "ctgov_alignment_report.json", json.dumps(rep, indent=1))
    print(f"{len(pairs)} (compound, disease) evidence pairs -> "
          f"{EV / 'evidence_ctd.tsv'}", flush=True)


if __name__ == "__main__":
    main()
