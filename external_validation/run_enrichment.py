"""E1 enrichment analysis: do KEEP sets enrich for post-snapshot evidence?

Implements external_validation/protocol_frozen.md §8 (2026-08-22, FROZEN).
Reads the exported full-space candidates and the fetched external evidence,
builds one compact numpy array per decision edge in a single streaming pass,
then evaluates every frozen analysis cell:

  spaces     : full | restricted (alignment-covered universe, per edge)
  thresholds : main (global calibrated 0.10) | t20 | relcond (per-relation)
  windows    : full | W1 2017-2020 | W2 2021-2026 (CtD: trial start years;
               CbG: document years; CbG additionally tier-1 <=10 nM vs
               tier-2 <=100 nM)

Statistics per cell: 2x2 contingency, one-sided Fisher exact (KEEP hits
more), hit rates with Wilson 95% CIs, lift, and 10 seeded random-size-matched
controls (seeds 1000-1009). Feasibility checkpoint: cells with < 50 evidence
hits in the space are flagged low-power instead of interpreted.

Outputs: results/external_validation_{ctd,cbg}.json

Usage: python external_validation/run_enrichment.py
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
from scipy.stats import fisher_exact

ROOT = Path(__file__).resolve().parent.parent
EV = ROOT / "external_validation"
RESULTS = ROOT / "results"
NODES_TSV = EV / "cache" / "ctgov" / "nodes.tsv"
RAND_SEEDS = list(range(1000, 1010))   # frozen (protocol §8)


# ---------- helpers ----------

def wilson(k: int, n: int, z: float = 1.959963985) -> tuple:
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def load_nodes() -> dict:
    """id -> name and per-kind int maps, from the cached canonical nodes.tsv."""
    id2name = {}
    gene_sym2id = {}
    with open(NODES_TSV) as f:
        header = f.readline()
        assert header.startswith("id\tname\tkind"), header
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 3:
                continue
            nid, name, kind = parts[0], parts[1], parts[2]
            id2name[nid] = name
            if kind == "Gene":
                gene_sym2id[name] = nid
    return {"id2name": id2name, "gene_sym2id": gene_sym2id}


class Interner:
    def __init__(self):
        self.d = {}

    def get(self, s: str) -> int:
        v = self.d.get(s)
        if v is None:
            v = len(self.d)
            self.d[s] = v
        return v


def stream_candidates(path: Path, key_of, hit_lookup, interner,
                       head_int=None, tail_int=None):
    """One pass -> (keys, keep flags, hit flags, heads, tails).

    heads/tails arrays are produced only when head_int/tail_int interners
    are supplied (single-pass mode for the large CbG file)."""
    H, T = [], []
    heads_l, tails_l = [], []
    keeps = {"main": [], "t20": [], "cost11": [], "relcond": []}
    hits = {"full": [], "W1": [], "W2": [], "tier1_full": [], "tier1_W1": [],
            "tier1_W2": []}
    with open(path) as f:
        header = f.readline().split("\t")
        assert header[0] == "head_label", header
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 9:
                continue
            h, t = parts[0], parts[2]
            k = key_of(h, t)
            hv = interner.get(k)
            if head_int is not None:
                heads_l.append(head_int.get(h))
                tails_l.append(tail_int.get(t))
            ev = hit_lookup(k)
            H.append(hv)
            T.append(1)
            keeps["main"].append(parts[5] == "1")
            keeps["t20"].append(parts[6] == "1")
            keeps["cost11"].append(parts[7] == "1")
            keeps["relcond"].append(parts[8] == "1")
            hits["full"].append(ev["full"])
            hits["W1"].append(ev["W1"])
            hits["W2"].append(ev["W2"])
            hits["tier1_full"].append(ev["tier1_full"])
            hits["tier1_W1"].append(ev["tier1_W1"])
            hits["tier1_W2"].append(ev["tier1_W2"])
    H = np.asarray(H, dtype=np.int64)
    del T
    keep_arr = {k: np.asarray(v, dtype=bool) for k, v in keeps.items()}
    hit_arr = {k: np.asarray(v, dtype=bool) for k, v in hits.items()}
    heads_a = (np.asarray(heads_l, dtype=np.int32)
               if head_int is not None else None)
    tails_a = (np.asarray(tails_l, dtype=np.int32)
               if tail_int is not None else None)
    return H, keep_arr, hit_arr, heads_a, tails_a


def cell_stats(space_mask, keep, hit, rand_rng):
    """Contingency + Fisher + lift + Wilson + 10 random matched controls."""
    n = int(space_mask.sum())
    if n == 0:
        return {"n_space": 0, "flag": "empty-space"}
    k = keep & space_mask
    w = ~keep & space_mask
    hk = int((k & hit).sum())
    nk = int(k.sum())
    hw = int((w & hit).sum())
    nw = int(w.sum())
    rate_k = hk / nk if nk else 0.0
    rate_w = hw / nw if nw else 0.0
    oddsr, p = fisher_exact([[hk, nk - hk], [hw, nw - hw]],
                            alternative="greater") if nk and nw else (float("nan"), float("nan"))
    lift = (rate_k / rate_w) if rate_w > 0 else (float("inf") if rate_k > 0
                                                 else float("nan"))
    rand_rates = []
    if nk:
        for sd in RAND_SEEDS:
            idx = rand_rng(n, size=nk) if False else \
                np.random.default_rng(sd).choice(np.flatnonzero(space_mask),
                                                 size=nk, replace=False)
            rand_rates.append(float(hit[idx].mean()))
    return {
        "n_space": n, "n_keep": nk, "n_withhold": nw,
        "keep_hits": hk, "withhold_hits": hw,
        "keep_rate": rate_k, "keep_rate_ci": wilson(hk, nk),
        "withhold_rate": rate_w, "withhold_rate_ci": wilson(hw, nw),
        "lift": lift,
        "fisher_p_one_sided": float(p) if p == p else None,
        "odds_ratio": float(oddsr) if oddsr == oddsr else None,
        "random_matched_rates": rand_rates,
        "low_power": int((hk + hw)) < 50,
    }


# ---------- CtD ----------

def analyze_ctd(nodes, suffix: str = "") -> dict:
    ev_path = EV / "evidence_ctd.tsv"
    with open(ev_path) as f:
        hdr = f.readline().split("\t")
        assert "start_years" in hdr, hdr
        ev = {}
        comp_with_ev, dis_with_ev = set(), set()
        for line in f:
            p = line.rstrip("\n").split("\t")
            if len(p) < 8:
                continue
            cid, did, years = p[0], p[2], p[5]
            if not cid.startswith("Compound::"):
                cid = "Compound::" + cid
            yrs = {int(y) for y in years.split(",") if y}
            ev[(cid, did)] = {
                "full": True,
                "W1": any(y <= 2020 for y in yrs),
                "W2": any(y >= 2021 for y in yrs),
                "tier1_full": True, "tier1_W1": any(y <= 2020 for y in yrs),
                "tier1_W2": any(y >= 2021 for y in yrs),
            }
            comp_with_ev.add(cid)
            dis_with_ev.add(did)

    intern = Interner()

    def key_of(h, t):
        return (h, t)

    def hit_lookup(k):
        return ev.get(k, {"full": False, "W1": False, "W2": False,
                          "tier1_full": False, "tier1_W1": False,
                          "tier1_W2": False})

    H, keep_arr, hit_arr, _, _ = stream_candidates(
        EV / f"candidates_ctd{suffix}.tsv", key_of, hit_lookup, intern)
    n = H.size
    # space masks
    full = np.ones(n, dtype=bool)
    # restricted: compounds and diseases that appear in >=1 evidence pair
    # (head/tail id interners built in the re-stream below)
    head_int, tail_int = Interner(), Interner()
    heads, tails = [], []
    with open(EV / f"candidates_ctd{suffix}.tsv") as f:
        f.readline()
        for line in f:
            p = line.split("\t", 3)
            heads.append(head_int.get(p[0]))
            tails.append(tail_int.get(p[1 + 1]))  # p[2]
    heads = np.asarray(heads)
    tails = np.asarray(tails)
    comp_ev_ids = {head_int.d[c] for c in comp_with_ev if c in head_int.d}
    dis_ev_ids = {tail_int.d[d] for d in dis_with_ev if d in tail_int.d}
    restricted = np.isin(heads, list(comp_ev_ids)) & np.isin(tails,
                                                             list(dis_ev_ids))
    rng = np.random.default_rng(0)
    out = {"n_evidence_pairs": len(ev),
           "n_space_full": int(full.sum()),
           "n_space_restricted": int(restricted.sum())}
    for space_name, space in (("full", full), ("restricted", restricted)):
        for th in ("main", "t20", "relcond", "cost11"):
            for win in ("full", "W1", "W2"):
                out[f"{space_name}|{th}|{win}"] = cell_stats(
                    space, keep_arr[th], hit_arr[win], rng)
    return out


# ---------- CbG ----------

def analyze_cbg(nodes, suffix: str = "") -> dict:
    gene_sym2id = nodes["gene_sym2id"]
    ev_path = EV / "evidence_cbg.tsv"
    ev = {}
    with open(ev_path) as f:
        hdr = f.readline().split("\t")
        assert "doc_years" in hdr, hdr
        for line in f:
            p = line.rstrip("\n").split("\t")
            if len(p) < 8:
                continue
            cid, sym, best, t1, years = p[0], p[1], float(p[2]), p[3] == "1", p[5]
            if not cid.startswith("Compound::"):
                cid = "Compound::" + cid
            gid = gene_sym2id.get(sym)
            if gid is None:
                continue
            yrs = {int(y) for y in years.split(",") if y}
            ev[(cid, gid)] = {
                "full": True,
                "W1": any(y <= 2020 for y in yrs),
                "W2": any(y >= 2021 for y in yrs),
                "tier1_full": t1,
                "tier1_W1": t1 and any(y <= 2020 for y in yrs),
                "tier1_W2": t1 and any(y >= 2021 for y in yrs),
            }
    mapped_comp = set()
    mc_path = EV / "chembl_mapped_compounds.tsv"
    if mc_path.exists():
        with open(mc_path) as f:
            f.readline()
            for line in f:
                mapped_comp.add(line.split("\t", 1)[0].strip())
    mapped_gene_ids = set()
    mg_path = EV / "chembl_mapped_genes.tsv"
    if mg_path.exists():
        with open(mg_path) as f:
            f.readline()
            for line in f:
                sym = line.split("\t", 1)[0].strip()
                gid = gene_sym2id.get(sym)
                if gid is not None:
                    mapped_gene_ids.add(gid)

    intern = Interner()

    def key_of(h, t):
        return (h, t)

    null_ev = {"full": False, "W1": False, "W2": False, "tier1_full": False,
               "tier1_W1": False, "tier1_W2": False}

    def hit_lookup(k):
        return ev.get(k, null_ev)

    head_int, tail_int = Interner(), Interner()
    H, keep_arr, hit_arr, heads, tails = stream_candidates(
        EV / f"candidates_cbg{suffix}.tsv", key_of, hit_lookup, intern,
        head_int=head_int, tail_int=tail_int)
    n = H.size
    comp_mapped_ids = {head_int.d[c] for c in mapped_comp if c in head_int.d}
    gene_mapped_ids = {tail_int.d[g] for g in mapped_gene_ids
                       if g in tail_int.d}
    restricted = np.isin(heads, list(comp_mapped_ids)) & np.isin(
        tails, list(gene_mapped_ids))
    rng = np.random.default_rng(0)
    out = {"n_evidence_pairs": len(ev),
           "n_space_full": int(n),
           "n_space_restricted": int(restricted.sum())}
    for space_name, space in (("full", np.ones(n, dtype=bool)),
                              ("restricted", restricted)):
        for th in ("main", "t20", "relcond"):
            for variant in ("full", "W1", "W2", "tier1_full", "tier1_W1",
                            "tier1_W2"):
                out[f"{space_name}|{th}|{variant}"] = cell_stats(
                    space, keep_arr[th], hit_arr[variant], rng)
    return out


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--suffix", default="",
                    help="candidates/output filename suffix, e.g. _complex")
    args = ap.parse_args()

    RESULTS.mkdir(exist_ok=True)
    nodes = load_nodes()
    for edge, fn in (("ctd", analyze_ctd), ("cbg", analyze_cbg)):
        print(f"=== {edge.upper()} ===", flush=True)
        res = fn(nodes, args.suffix)
        out_path = RESULTS / f"external_validation_{edge}{args.suffix}.json"
        out_path.write_text(json.dumps(res, indent=1))
        # console digest: the frozen primary cells
        prim = (res.get("full|main|full") if edge == "ctd"
                else res.get("full|main|tier1_full"))
        print(json.dumps(prim, indent=1), flush=True)
        print(f"-> {out_path}", flush=True)


if __name__ == "__main__":
    main()
