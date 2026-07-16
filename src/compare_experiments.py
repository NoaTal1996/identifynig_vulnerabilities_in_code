"""
Pairwise statistical comparison of two evaluations on the same test set.

Consumes the *_predictions.jsonl files produced by train.py / eval.py and
runs two paired tests:
  1. McNemar's test on correct/wrong per example (for accuracy).
  2. Paired percentile bootstrap of the metric difference (for F1 / AUC /
     accuracy / precision / recall).

Pairing rule (same-test-set requirement):
  - If both files have non-empty `sample_id` for every row, pairs are matched
    by sample_id (order-independent).
  - Otherwise, if both files have the same length, pairs are matched by row
    index (assumes identical test-set order — this holds for the HF-dataset
    path because get_dataset() is deterministic).
  - Otherwise, the script refuses to compare and asks for sample_ids.

Usage:
  python3 src/compare_experiments.py \
      --a results/juliet_source_metrics_predictions.jsonl \
      --b results/E4_M1_on_llm_juliet_source_predictions.jsonl \
      --output_file results/compare_M1_E1_vs_E4.json
"""
import argparse
import json
import math
import os

import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score


def load_predictions(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def canonicalize_juliet_id(sid):
    """Bring Juliet sample_ids from HF path and LLM-JSONL path into one form.

    HF-path (data_loader.get_dataset)        : 'CWE127_Buffer_Underread__malloc_char_loop_11-good'
    JSONL (generate_llm_rewrites)            : 'cwe127_buffer_underread__malloc_char_loop_11.c::good'

    Canonical form                           : 'cwe127_buffer_underread__malloc_char_loop_11:good'
    """
    if not isinstance(sid, str):
        return sid
    s = sid.strip().lower()
    for suffix in (".cpp", ".c"):
        s = s.replace(suffix + "::", ":")
    s = s.replace("::", ":").replace("-good", ":good").replace("-bad", ":bad")
    return s


def match_pairs(a_rows, b_rows):
    """Return (labels_a, preds_a, probs_a, labels_b, preds_b, probs_b, mode).

    Uses sample_id matching when both sides have IDs for every row; otherwise
    falls back to positional pairing when lengths match. Raises otherwise.
    """
    def has_all_ids(rows):
        return all(r.get("sample_id") for r in rows)

    if has_all_ids(a_rows) and has_all_ids(b_rows):
        by_id_a = {r["sample_id"]: r for r in a_rows}
        by_id_b = {r["sample_id"]: r for r in b_rows}
        common_ids = sorted(set(by_id_a) & set(by_id_b))
        if not common_ids:
            raise ValueError("No sample_ids overlap between the two files.")
        a = [by_id_a[i] for i in common_ids]
        b = [by_id_b[i] for i in common_ids]
        mode = f"sample_id (n_common={len(common_ids)}, dropped_a={len(a_rows)-len(common_ids)}, dropped_b={len(b_rows)-len(common_ids)})"
    else:
        if len(a_rows) != len(b_rows):
            raise ValueError(
                f"Cannot pair: sample_id missing on one side and lengths differ "
                f"(a={len(a_rows)}, b={len(b_rows)}). Re-run eval so predictions include sample_id."
            )
        # Sanity: labels must agree row-by-row for the pairing to be valid.
        mismatches = sum(1 for x, y in zip(a_rows, b_rows) if int(x["label"]) != int(y["label"]))
        if mismatches > 0:
            raise ValueError(
                f"Positional pairing rejected: {mismatches} rows have different labels between the two files. "
                f"This means the two evaluations are not on the same test set in the same order."
            )
        a, b = a_rows, b_rows
        mode = f"positional (n={len(a)})"

    labels_a = np.asarray([int(r["label"]) for r in a])
    preds_a = np.asarray([int(r["pred"]) for r in a])
    probs_a = np.asarray([float(r["prob_class1"]) for r in a])
    labels_b = np.asarray([int(r["label"]) for r in b])
    preds_b = np.asarray([int(r["pred"]) for r in b])
    probs_b = np.asarray([float(r["prob_class1"]) for r in b])

    if not np.array_equal(labels_a, labels_b):
        raise ValueError("Paired labels do not agree; the two evaluations must be on the same test set.")

    return labels_a, preds_a, probs_a, preds_b, probs_b, mode


def mcnemar(preds_a, preds_b, labels):
    """Exact-mid-p McNemar with continuity correction for large N.

    Returns (b, c, statistic, p_value) where:
      b = # examples A correct, B wrong
      c = # examples A wrong, B correct
    """
    correct_a = preds_a == labels
    correct_b = preds_b == labels
    b = int(np.sum(correct_a & ~correct_b))
    c = int(np.sum(~correct_a & correct_b))
    n = b + c
    if n == 0:
        return b, c, 0.0, 1.0

    # For small n (<25) use exact binomial two-sided; else chi-square with continuity correction.
    if n < 25:
        # Exact two-sided binomial test at k=min(b,c), p=0.5, N=n.
        from math import comb
        k = min(b, c)
        p_two = 2 * sum(comb(n, i) for i in range(k + 1)) / (2 ** n)
        p_two = min(1.0, p_two)
        return b, c, float(k), float(p_two)

    stat = (abs(b - c) - 1) ** 2 / (b + c)
    # Chi-square with 1 dof survival function via error function on the corresponding normal.
    # P(Chi^2_1 >= stat) = erfc(sqrt(stat/2))
    p = math.erfc(math.sqrt(stat / 2.0))
    return b, c, float(stat), float(p)


def _metric(y_true, y_pred, y_prob, name):
    if name == "accuracy":
        return accuracy_score(y_true, y_pred)
    if name in ("f1", "precision", "recall"):
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true, y_pred, average="binary", zero_division=0
        )
        return {"precision": precision, "recall": recall, "f1": f1}[name]
    if name == "auc":
        try:
            return roc_auc_score(y_true, y_prob)
        except ValueError:
            return float("nan")
    raise ValueError(name)


def paired_bootstrap_diff(labels, preds_a, probs_a, preds_b, probs_b,
                          n_boot=1000, seed=42, alpha=0.05):
    """Bootstrap the paired difference metric(B) - metric(A) for each metric."""
    labels = np.asarray(labels)
    n = len(labels)
    rng = np.random.default_rng(seed)
    metrics = ["accuracy", "f1", "precision", "recall", "auc"]

    point = {}
    for m in metrics:
        point[m] = {
            "a": float(_metric(labels, preds_a, probs_a, m)),
            "b": float(_metric(labels, preds_b, probs_b, m)),
        }
        point[m]["diff_b_minus_a"] = point[m]["b"] - point[m]["a"]

    diffs = {m: [] for m in metrics}
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        yt = labels[idx]
        pa, pb = preds_a[idx], preds_b[idx]
        qa, qb = probs_a[idx], probs_b[idx]
        for m in metrics:
            va = _metric(yt, pa, qa, m)
            vb = _metric(yt, pb, qb, m)
            diffs[m].append(vb - va)

    lo_p, hi_p = 100 * (alpha / 2), 100 * (1 - alpha / 2)
    out = {}
    for m in metrics:
        arr = np.asarray(diffs[m], dtype=float)
        finite = arr[np.isfinite(arr)]
        if finite.size == 0:
            out[m] = {**point[m], "ci_low": float("nan"), "ci_high": float("nan"),
                      "p_two_sided": float("nan"), "n_boot": 0}
            continue
        lo = float(np.percentile(finite, lo_p))
        hi = float(np.percentile(finite, hi_p))
        # Two-sided p-value: 2 * min(P(diff<=0), P(diff>=0)).
        p_left = float(np.mean(finite <= 0))
        p_right = float(np.mean(finite >= 0))
        p_two = float(min(1.0, 2 * min(p_left, p_right)))
        out[m] = {
            **point[m],
            "ci_low": lo,
            "ci_high": hi,
            "p_two_sided": p_two,
            "n_boot": int(finite.size),
            "significant_at_95": bool((lo > 0) or (hi < 0)),
        }
    return out


def main():
    parser = argparse.ArgumentParser(description="Compare two per-example prediction files on the same test set.")
    parser.add_argument("--a", required=True, help="Path to first predictions JSONL (baseline)")
    parser.add_argument("--b", required=True, help="Path to second predictions JSONL (candidate)")
    parser.add_argument("--label_a", default="A", help="Human label for --a")
    parser.add_argument("--label_b", default="B", help="Human label for --b")
    parser.add_argument("--n_boot", type=int, default=1000)
    parser.add_argument("--boot_seed", type=int, default=42)
    parser.add_argument("--output_file", help="Optional JSON output path.")
    parser.add_argument("--canonicalize_juliet_ids", action="store_true",
                        help="Normalize Juliet sample_ids (HF path vs LLM-JSONL) before matching.")
    args = parser.parse_args()

    print(f"=== Pairwise comparison: {args.label_a}  vs  {args.label_b} ===")
    print(f"  A: {args.a}")
    print(f"  B: {args.b}")

    a_rows = load_predictions(args.a)
    b_rows = load_predictions(args.b)
    if args.canonicalize_juliet_ids:
        for r in a_rows:
            r["sample_id"] = canonicalize_juliet_id(r.get("sample_id", ""))
        for r in b_rows:
            r["sample_id"] = canonicalize_juliet_id(r.get("sample_id", ""))
    labels, preds_a, probs_a, preds_b, probs_b, mode = match_pairs(a_rows, b_rows)
    n = len(labels)
    print(f"  Paired via: {mode}")
    print(f"  N paired examples: {n}")

    # McNemar
    b_cnt, c_cnt, stat, p = mcnemar(preds_a, preds_b, labels)
    print(f"\n--- McNemar (accuracy comparison) ---")
    print(f"  b (A correct, B wrong): {b_cnt}")
    print(f"  c (A wrong,  B correct): {c_cnt}")
    print(f"  statistic: {stat:.4f}")
    print(f"  p-value  : {p:.4g}")
    verdict = "significant" if p < 0.05 else "not significant"
    print(f"  verdict  : {verdict} at alpha=0.05")

    # Paired bootstrap of the difference for every metric
    print(f"\n--- Paired bootstrap of metric differences (n_boot={args.n_boot}) ---")
    boot = paired_bootstrap_diff(
        labels, preds_a, probs_a, preds_b, probs_b,
        n_boot=args.n_boot, seed=args.boot_seed,
    )
    print(f"  {'metric':<10} {'A':>8} {'B':>8} {'B-A':>8}   95% CI of (B-A)          p_2s   sig?")
    for m in ["accuracy", "f1", "precision", "recall", "auc"]:
        r = boot[m]
        sig = "***" if r["significant_at_95"] else ""
        print(f"  {m:<10} {r['a']:>8.4f} {r['b']:>8.4f} {r['diff_b_minus_a']:>+8.4f}   "
              f"[{r['ci_low']:>+.4f}, {r['ci_high']:>+.4f}]   {r['p_two_sided']:>.4g}  {sig}")

    out = {
        "a": args.a,
        "b": args.b,
        "label_a": args.label_a,
        "label_b": args.label_b,
        "pairing_mode": mode,
        "n_paired": int(n),
        "mcnemar": {"b": b_cnt, "c": c_cnt, "statistic": stat, "p_value": p,
                    "significant_at_95": bool(p < 0.05)},
        "bootstrap_diff": boot,
        "n_boot": int(args.n_boot),
        "boot_seed": int(args.boot_seed),
    }
    if args.output_file:
        os.makedirs(os.path.dirname(os.path.abspath(args.output_file)), exist_ok=True)
        with open(args.output_file, "w") as f:
            json.dump(out, f, indent=4)
        print(f"\nSaved results to {args.output_file}")


if __name__ == "__main__":
    main()
