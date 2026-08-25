"""Fetch post-2017 ChEMBL bioactivity evidence for CbG external validation.

Implements external_validation/protocol_frozen.md §6 (2026-08-22, FROZEN):
  * Compound alignment: hetionet compounds.tsv InChIKey -> ChEMBL molecule
    (exact structural key match; no fuzzy matching);
  * Target alignment: ChEMBL SINGLE PROTEIN targets -> UniProt accessions ->
    gene primary symbols -> exact match against Hetionet Gene names;
  * Activities: standard_type in {IC50, Ki, Kd, EC50, AC50, Potency},
    standard_relation '=' and standard_value <= 100 nM (Tier-2; Tier-1 <=10 nM
    derived at assembly), organism unrestricted (frozen);
  * Date: activity counts as post-snapshot evidence iff document_year > 2017
    (client-side filter - the webresource 'activity_year__gte' parameter was
    verified 2026-08-22 to be silently ignored and MUST NOT be relied on);
  * A filter self-check aborts loudly if the server-side value/type filters
    appear to be ignored (defense against another silent-filter trap).

Every network response is cached under external_validation/cache/chembl/ by
SHA-256 of the URL, so re-runs resume. All writes are confined to
external_validation/. Host allowlist: www.ebi.ac.uk, rest.uniprot.org,
raw.githubusercontent.com.

Usage: python external_validation/fetch_chembl.py [--delay 0.3]
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
CACHE = EV / "cache" / "chembl"
CTGOV_CACHE = EV / "cache" / "ctgov"
ALLOWED_HOSTS = {"www.ebi.ac.uk", "rest.uniprot.org",
                 "raw.githubusercontent.com"}
CHEMBL = "https://www.ebi.ac.uk/chembl/api/data"
UNIPROT = "https://rest.uniprot.org"
NODES_URL = ("https://raw.githubusercontent.com/hetio/hetionet/main/"
             "hetnet/tsv/hetionet-v1.0-nodes.tsv")
# Compound InChIKeys were extracted once from the canonical
# hetnet/json/hetionet-v1.0.json.bz2 (Git LFS) into this local cache file.
INCHIKEY_CACHE = CACHE / "compound_inchikeys.json"
STANDARD_TYPES = ["IC50", "Ki", "Kd", "EC50", "AC50", "Potency"]
VALUE_MAX_NM = 100          # frozen Tier-2 upper bound
DOC_YEAR_MIN = 2018         # frozen: document_year > 2017


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


def sha_name(url: str) -> str:
    return hashlib.sha256(url.encode("utf-8")).hexdigest()


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
    def redirect_request(self, req, fp, code, msg, headers, newurl):  # noqa: D102
        _assert_safe_url(newurl)
        return super().redirect_request(req, fp, code, msg, headers, newurl)


_OPENER = urllib.request.build_opener(_SafeRedirectHandler())


def safe_get(url: str, retries: int = 5) -> bytes:
    for i in range(retries):
        _assert_safe_url(url)
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "..."} )
            with _OPENER.open(req, timeout=90) as r:
                return r.read()
        except Exception:  # noqa: BLE001 - network retry loop
            if i == retries - 1:
                raise
            time.sleep(5 * (i + 1))
    raise RuntimeError("unreachable")


def cached_json(url: str, delay: float) -> dict:
    rel = CACHE / f"{sha_name(url)}.json"
    if exists_under(rel):
        return json.loads(read_under(rel))
    data = json.loads(safe_get(url))
    write_under(rel, json.dumps(data))
    time.sleep(delay)
    return data


# ---------- step 1: hetionet compound InChIKeys ----------

def load_compound_inchikeys(delay: float) -> dict:
    """Compound id -> bare InChIKey from the local extraction of the canonical
    hetionet-v1.0 JSON (see INCHIKEY_CACHE note above). The JSON stores keys
    as 'InChIKey=<key>'; the prefix is stripped here."""
    if not exists_under(INCHIKEY_CACHE):
        raise RuntimeError(
            f"missing {INCHIKEY_CACHE}: extract it once from "
            "hetnet/json/hetionet-v1.0.json.bz2 (see qa/02 experiment log)")
    raw = json.loads(read_under(INCHIKEY_CACHE))
    out = {}
    for cid, ik in raw.items():
        ik = ik.strip()
        if ik.upper().startswith("INCHIKEY="):
            ik = ik.split("=", 1)[1].strip()
        cid = cid if cid.startswith("Compound::") else f"Compound::{cid}"
        if ik:
            out[cid] = ik
    return out


# ---------- step 2: InChIKey -> ChEMBL molecule ----------

def map_compounds_to_chembl(inchikeys: dict, delay: float) -> dict:
    out, misses = {}, 0
    for i, (cid, ik) in enumerate(sorted(inchikeys.items())):
        url = (f"{CHEMBL}/molecule.json?molecule_structures__standard_inchi_key="
               f"{urllib.parse.quote(ik)}")
        d = cached_json(url, delay)
        mols = d.get("molecules", [])
        hit = None
        for m in mols:
            structs = m.get("molecule_structures") or {}
            if structs.get("standard_inchi_key") == ik:
                hit = m.get("molecule_chembl_id")
                break
        if hit:
            out[cid] = hit
        else:
            misses += 1
        if (i + 1) % 200 == 0:
            print(f"  compounds {i+1}/{len(inchikeys)} mapped={len(out)} "
                  f"miss={misses}", flush=True)
    print(f"[compounds] mapped {len(out)}/{len(inchikeys)} (misses {misses})",
          flush=True)
    return out


# ---------- step 3: ChEMBL single-protein targets ----------

def fetch_targets(delay: float) -> list:
    targets, offset = [], 0
    while True:
        url = (f"{CHEMBL}/target.json?target_type=SINGLE%20PROTEIN"
               f"&limit=100&offset={offset}")
        d = cached_json(url, delay)
        page = d.get("targets", [])
        targets.extend(page)
        total = d.get("page_meta", {}).get("total_count", len(targets))
        offset += len(page)
        if not page or offset >= total or offset > 20000:
            break
    print(f"[targets] {len(targets)} SINGLE PROTEIN targets", flush=True)
    return targets


# ---------- step 4: UniProt -> gene symbol ----------

def uniprot_gene_symbols(accessions: list, delay: float) -> dict:
    acc2sym, i = {}, 0
    for j in range(0, len(accessions), 100):
        batch = accessions[j:j + 100]
        q = " OR ".join(f"accession:{a}" for a in batch)
        url = (f"{UNIPROT}/uniprotkb/search?query={urllib.parse.quote(q)}"
               f"&fields=accession,gene_primary&format=tsv&size=500")
        rel = CACHE / f"{sha_name(url)}.tsv"
        if exists_under(rel):
            text = read_under(rel)
        else:
            text = safe_get(url).decode("utf-8")
            write_under(rel, text)
            time.sleep(delay)
        lines = [l for l in text.splitlines() if l and not l.startswith(
            "Entry")]
        for line in lines:
            parts = line.split("\t")
            if len(parts) >= 2 and parts[1].strip():
                first_gene = parts[1].split(";")[0].strip()
                acc2sym[parts[0].strip()] = first_gene
        i += len(batch)
        if i % 1000 < 100:
            print(f"  uniprot {i}/{len(accessions)}", flush=True)
    print(f"[uniprot] {len(acc2sym)} accession->symbol mappings", flush=True)
    return acc2sym


# ---------- step 5: potent activities per compound batch ----------

def fetch_activities(mol_chembl_ids: list, delay: float) -> list:
    rows = []
    type_q = ",".join(STANDARD_TYPES)
    n_batches = (len(mol_chembl_ids) + 39) // 40
    for bi in range(n_batches):
        batch = mol_chembl_ids[bi * 40:(bi + 1) * 40]
        mol_q = ",".join(batch)
        offset = 0
        while True:
            url = (f"{CHEMBL}/activity.json?molecule_chembl_id__in={mol_q}"
                   f"&standard_type__in={type_q}"
                   f"&standard_value__lte={VALUE_MAX_NM}"
                   f"&standard_relation=%3D"
                   f"&limit=1000&offset={offset}")
            d = cached_json(url, delay)
            acts = d.get("activities", [])
            # --- filter self-check (protocol: no silently-ignored filters) ---
            for a in acts[:50]:
                v = a.get("standard_value")
                if v is not None and float(v) > VALUE_MAX_NM:
                    raise RuntimeError(
                        "standard_value__lte filter appears IGNORED by the "
                        "API (got value >100): refusing to continue with an "
                        "unfiltered fetch; switch to client-side filtering "
                        "with an explicit protocol amendment.")
            for a in acts:
                rows.append({
                    "molecule": a.get("molecule_chembl_id"),
                    "target": a.get("target_chembl_id"),
                    "type": a.get("standard_type"),
                    "value": float(a["standard_value"])
                             if a.get("standard_value") else None,
                    "units": a.get("standard_units"),
                    "relation": a.get("standard_relation"),
                    "doc_year": a.get("document_year"),
                })
            offset += len(acts)
            total = d.get("page_meta", {}).get("total_count", 0)
            if not acts or (total and offset >= total) or offset > 50000:
                break
        print(f"  activity batch {bi+1}/{n_batches}: {len(rows):,} rows",
              flush=True)
    return rows


# ---------- assembly ----------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--delay", type=float, default=0.3)
    args = ap.parse_args()

    CACHE.mkdir(parents=True, exist_ok=True)
    report = {}

    # 1-2: compounds
    iks = load_compound_inchikeys(args.delay)
    report["n_hetionet_compounds_with_inchikey"] = len(iks)
    comp_map = map_compounds_to_chembl(iks, args.delay)
    report["n_compounds_mapped_to_chembl"] = len(comp_map)

    # 3: targets
    targets = fetch_targets(args.delay)
    report["n_single_protein_targets"] = len(targets)

    # 4: uniprot -> symbol; keep targets whose symbol is a Hetionet Gene
    nodes_rel = CTGOV_CACHE / "nodes.tsv"
    if not exists_under(nodes_rel):
        nodes_rel = CACHE / "nodes.tsv"
    gene_names = set()
    for line in read_under(nodes_rel).splitlines()[1:]:
        parts = line.split("\t")
        if len(parts) >= 3 and parts[2] == "Gene":
            gene_names.add(parts[1])
    report["n_hetionet_genes"] = len(gene_names)
    accs = sorted({c["accession"] for t in targets
                   for c in t.get("target_components", []) if c.get("accession")})
    acc2sym = uniprot_gene_symbols(accs, args.delay)
    sym_targets = {}  # chembl_target_id -> gene symbol
    for t in targets:
        syms = {acc2sym.get(c.get("accession")) for c in
                t.get("target_components", [])} - {None}
        hits = syms & gene_names
        if len(hits) == 1:
            sym_targets[t["target_chembl_id"]] = hits.pop()
        elif len(hits) > 1:
            report.setdefault("multi_symbol_targets", 0)
            report["multi_symbol_targets"] += 1
    report["n_targets_mapped_to_hetionet_genes"] = len(sym_targets)

    # 5: activities for mapped compounds
    acts = fetch_activities(sorted(comp_map.values()), args.delay)
    report["n_activity_rows_fetched"] = len(acts)

    # 6: assemble post-2017 potent binding pairs
    inv_comp = {v: k for k, v in comp_map.items()}
    pairs = {}
    kept = 0
    for a in acts:
        if (a["relation"] != "=" or a["units"] != "nM"
                or a["value"] is None or a["value"] > VALUE_MAX_NM
                or a["doc_year"] is None or int(a["doc_year"]) < DOC_YEAR_MIN):
            continue
        cid = inv_comp.get(a["molecule"])
        sym = sym_targets.get(a["target"])
        if not cid or not sym:
            continue
        kept += 1
        rec = pairs.setdefault((cid, sym), {"years": set(), "best_nM": None,
                                            "types": set()})
        rec["years"].add(int(a["doc_year"]))
        rec["types"].add(a["type"])
        if rec["best_nM"] is None or a["value"] < rec["best_nM"]:
            rec["best_nM"] = a["value"]
    report["n_post2017_potent_activity_rows"] = kept
    report["n_cbgene_evidence_pairs"] = len(pairs)

    lines = ["compound_id\tgene_symbol\tbest_value_nM\ttier1_best_nM\t"
             "types\tdoc_years\tearliest_year\tlatest_year"]
    for (cid, sym), rec in sorted(pairs.items()):
        t1 = "1" if rec["best_nM"] is not None and rec["best_nM"] <= 10 else "0"
        yrs = sorted(rec["years"])
        lines.append(f"{cid}\t{sym}\t{rec['best_nM']}\t{t1}\t"
                     f"{','.join(sorted(rec['types']))}\t"
                     f"{','.join(str(y) for y in yrs)}\t{yrs[0]}\t{yrs[-1]}")
    write_under(EV / "evidence_cbg.tsv", "\n".join(lines) + "\n")
    write_under(EV / "chembl_alignment_report.json", json.dumps(report,
                                                                indent=1))
    # aligned-universe exports for the restricted sensitivity analyses
    mc = ["compound_id\tchembl_id"]
    for cid, cmbl in sorted(comp_map.items()):
        if not cid.startswith("Compound::"):
            mc.append(f"Compound::{cid}\t{cmbl}")
        else:
            mc.append(f"{cid}\t{cmbl}")
    write_under(EV / "chembl_mapped_compounds.tsv", "\n".join(mc) + "\n")
    gt = ["gene_symbol\tchembl_target_id"]
    for tid, sym in sorted(sym_targets.items(), key=lambda kv: kv[1]):
        gt.append(f"{sym}\t{tid}")
    write_under(EV / "chembl_mapped_genes.tsv", "\n".join(gt) + "\n")
    print("ALIGNMENT REPORT:", json.dumps(report, indent=1), flush=True)
    print(f"{len(pairs)} (compound, gene) evidence pairs -> "
          f"{EV / 'evidence_cbg.tsv'}", flush=True)


if __name__ == "__main__":
    main()
