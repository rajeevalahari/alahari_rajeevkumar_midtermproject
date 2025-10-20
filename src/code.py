#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import time
from pathlib import Path
from itertools import combinations
import pandas as pd

# importing mlxtend; if missing, give a helpful message without crashing.
try:
    from mlxtend.preprocessing import TransactionEncoder
    from mlxtend.frequent_patterns import apriori, association_rules, fpgrowth
except Exception:
    print(
        "\n[!] The 'mlxtend' package is required.\n"
        "    Install it first:\n"
        "        pip install mlxtend\n"
    )
    raise

# -----------------------------
# Configuration & Data Loading
# -----------------------------

CANDIDATE_FILES = {
    1: ["Amazon_Transactions.csv"],
    2: ["BestBuy_Transactions.csv"],
    3: ["KMart_Transactions.csv"],
    4: ["Generic_Transactions.csv"],
    5: ["Nike_Transactions.csv"],
}

MENU = (
    "Select your Dataset:\n"
    "  1. Amazon\n"
    "  2. BestBuy\n"
    "  3. K-Mart\n"
    "  4. Generic\n"
    "  5. Nike\n"
    "  0. Exit\n"
    "Enter choice (0-5): "
)

SCRIPT_DIR = Path(__file__).parent if "__file__" in globals() else Path.cwd()
PROJECT_ROOT = SCRIPT_DIR.parent
DATA_DIR = PROJECT_ROOT / "data"

def print_header(title: str):
    bar = "=" * len(title)
    print(f"\n{title}\n{bar}")

def find_existing_path(candidates):
    """Return the first existing file path from the candidates list."""
    for name in candidates:
        p = DATA_DIR / name
        if p.exists():
            return p
    return None

def read_csv_safely(path: Path) -> pd.DataFrame:
    """Read CSV with sensible defaults and fallback encoding."""
    try:
        return pd.read_csv(path)
    except UnicodeDecodeError:
        return pd.read_csv(path, encoding="latin-1")

# -----------------------------
# Transaction Extraction
# -----------------------------

LIKELY_BASKET_COLS = ["Transaction", "Items", "ItemList", "Basket", "Products"]
ID_COL_CANDIDATES = ["TransactionID", "TransID", "InvoiceNo", "OrderID", "TID", "BasketID", "TxnID"]
ITEM_COL_CANDIDATES = ["Item", "Items", "Product", "Description", "SKU"]
DELIMS = [",", ";", "|"]

def is_delimited_basket_series(series: pd.Series) -> bool:
    """Heuristic: a string column where many rows contain a delimiter."""
    if series.dtype == object:
        sample = series.dropna().astype(str).head(200)
        if sample.empty:
            return False
        hits = sum(1 for val in sample if any(d in val for d in DELIMS))
        return hits >= max(5, len(sample) // 5)
    return False

def extract_transactions(df: pd.DataFrame):
    """
    Return list[list[str]] transactions from:
    1) One row per transaction, with a delimited items column
    2) One row per (transaction_id, item), grouped
    """
    # 1) Direct basket column
    for col in LIKELY_BASKET_COLS:
        if col in df.columns and is_delimited_basket_series(df[col]):
            baskets = []
            for raw in df[col].fillna(""):
                s = str(raw)
                delim = max(DELIMS, key=lambda d: s.count(d))
                parts = [p.strip() for p in s.split(delim) if p.strip()]
                if parts:
                    baskets.append(parts)
            if baskets:
                return baskets

    # 2) (transaction_id, item)
    id_col = next((c for c in ID_COL_CANDIDATES if c in df.columns), None)
    item_col = next((c for c in ITEM_COL_CANDIDATES if c in df.columns), None)
    if id_col and item_col:
        grouped = (
            df[[id_col, item_col]]
            .dropna()
            .astype({id_col: str, item_col: str})
            .groupby(id_col)[item_col]
            .apply(lambda s: [x.strip() for x in s if str(x).strip()])
        )
        baskets = [lst for lst in grouped.tolist() if lst]
        if baskets:
            return baskets

    # 3) Last resort: any object column that looks delimited
    for col in df.columns:
        if df[col].dtype == object and is_delimited_basket_series(df[col]):
            baskets = []
            for raw in df[col].fillna(""):
                s = str(raw)
                delim = max(DELIMS, key=lambda d: s.count(d))
                parts = [p.strip() for p in s.split(delim) if p.strip()]
                if parts:
                    baskets.append(parts)
            if baskets:
                return baskets

    raise ValueError(
        "Could not detect transactions. Use a delimited items column "
        "(e.g., 'Items' = 'A,B,C') OR a pair like (TransactionID, Item)."
    )

# -----------------------------
# Brute-force Apriori (baseline)
# -----------------------------

def brute_force_apriori(transactions, min_support: float):
    """
    Simple Apriori-like baseline using combinations over unique items.
    Returns frequent_patterns (list[tuple]) and pattern_counts (list[int]).
    """
    unique_items = sorted({itm for tx in transactions for itm in tx})
    n_tx = len(transactions)

    def support_count(itemset):
        s = set(itemset)
        return sum(1 for tx in transactions if s.issubset(tx))

    k = 1
    current_items = unique_items[:]
    frequent_patterns, pattern_counts = [], []

    while current_items:
        candidate_itemsets = list(combinations(current_items, k))
        new_freq = []
        for cand in candidate_itemsets:
            cnt = support_count(cand)
            if cnt >= min_support * n_tx:
                new_freq.append(cand)
                frequent_patterns.append(cand)
                pattern_counts.append(cnt)
        items_in_new = sorted({i for it in new_freq for i in it})
        k += 1
        current_items = items_in_new

    return frequent_patterns, pattern_counts

def generate_rules_from_patterns(frequent_patterns, pattern_counts, n_tx, min_conf: float):
    """
    Rule generation from frequent patterns.
    Returns list of tuples: (antecedent, consequent, support, confidence)
    """
    rules = []
    supp_map = {tuple(sorted(p)): c / n_tx for p, c in zip(frequent_patterns, pattern_counts)}

    for pattern, supp_cnt in zip(frequent_patterns, pattern_counts):
        if len(pattern) < 2:
            continue
        pat_set = set(pattern)
        for r in range(1, len(pattern)):
            for antecedent in combinations(pattern, r):
                consequent = tuple(sorted(pat_set - set(antecedent)))
                antecedent = tuple(sorted(antecedent))
                supp_XY = supp_map.get(tuple(sorted(pattern)), supp_cnt / n_tx)
                supp_X = supp_map.get(antecedent, None)
                if not supp_X:
                    continue
                conf = supp_XY / supp_X
                if conf + 1e-12 >= min_conf:
                    rules.append((antecedent, consequent, supp_XY, conf))
    # Sort by confidence desc, then support desc
    rules.sort(key=lambda x: (x[3], x[2]), reverse=True)
    return rules

# -----------------------------
# Pretty Printing (print ALL)
# -----------------------------

def print_frequent_patterns(frequent_patterns, pattern_counts, n_tx):
    print_header("Frequent Patterns")
    rows = []
    for pat, cnt in zip(frequent_patterns, pattern_counts):
        supp = cnt / n_tx
        rows.append((pat, supp))
    rows.sort(key=lambda x: (x[1], len(x[0])), reverse=True)
    if not rows:
        print("(none)")
        return
    for i, (pat, supp) in enumerate(rows, start=1):
        print(f"[{i:>3}] {list(pat)} | support={supp:.4f}")

def print_rules(rules):
    print_header("Association Rules")
    if not rules:
        print("(no rules met the thresholds)")
        return
    for i, (ante, cons, supp, conf) in enumerate(rules, start=1):
        print(
            f"[{i:>3}] {list(ante)}  ->  {list(cons)}  "
            f"| support={supp:.4f}, confidence={conf:.4f}"
        )

# -----------------------------
# mlxtend Pipelines
# -----------------------------

def encode_transactions(transactions):
    te = TransactionEncoder()
    arr = te.fit(transactions).transform(transactions)
    df_enc = pd.DataFrame(arr, columns=te.columns_)
    return df_enc

def run_mlxtend_apriori(df_enc, min_support, min_confidence):
    start = time.time()
    fi = apriori(df_enc, min_support=min_support, use_colnames=True)
    rules = association_rules(fi, metric="confidence", min_threshold=min_confidence)
    runtime = time.time() - start
    return fi, rules, runtime

def run_mlxtend_fpgrowth(df_enc, min_support, min_confidence):
    start = time.time()
    fi = fpgrowth(df_enc, min_support=min_support, use_colnames=True)
    rules = association_rules(fi, metric="confidence", min_threshold=min_confidence)
    runtime = time.time() - start
    return fi, rules, runtime

def print_mlxtend_outputs(name, fi, rules):
    # ---- Frequent Itemsets ----
    print_header(f"{name} — Frequent Itemsets")
    if fi.empty:
        print("(none)")
    else:
        fi2 = fi.copy()
        fi2["k"] = fi2["itemsets"].map(len)  # ensure sort key is a column
        fi_sorted = fi2.sort_values(["support", "k"], ascending=[False, False])
        for i, row in enumerate(fi_sorted.itertuples(index=False), start=1):
            print(f"[{i:>3}] {sorted(list(row.itemsets))} | support={row.support:.4f}")

    # ---- Rules (support & confidence only) ----
    print_header(f"{name} — Association Rules")
    if rules.empty:
        print("(none)")
    else:
        # Sort by confidence then support if present
        sort_cols = [c for c in ["confidence", "support"] if c in rules.columns]
        rules_sorted = rules.sort_values(sort_cols, ascending=[False, False]) if sort_cols else rules
        for i, row in enumerate(rules_sorted.itertuples(index=False), start=1):
            ants = sorted(list(getattr(row, "antecedents", [])))
            cons = sorted(list(getattr(row, "consequents", [])))
            support = getattr(row, "support", float("nan"))
            conf = getattr(row, "confidence", float("nan"))
            print(
                f"[{i:>3}] {ants} -> {cons}  "
                f"| support={support:.4f}, confidence={conf:.4f}"
            )

# -----------------------------
# Defensive Input Helpers
# -----------------------------

def ask_menu_choice() -> int:
    """Ask for a dataset choice (0–5). Loops until valid."""
    while True:
        raw = input(MENU).strip()
        if raw == "":
            print("Please enter a number between 0 and 5.")
            continue
        if not raw.isdigit():
            print("Invalid input. Enter a number between 0 and 5.")
            continue
        val = int(raw)
        if 0 <= val <= 5:
            return val
        print("Out of range. Enter a number between 0 and 5.")

def ask_float(prompt: str, lo=0.0, hi=1.0) -> float:
    """Ask for a float in [lo, hi]. Loops until valid."""
    while True:
        raw = input(prompt).strip()
        if raw == "":
            print(f"Please enter a value between {lo} and {hi}.")
            continue
        try:
            if raw.startswith("."):
                raw = "0" + raw
            val = float(raw)
        except ValueError:
            print("Invalid number. Try again.")
            continue
        if lo <= val <= hi:
            return val
        print(f"Out of range. Enter a value between {lo} and {hi}.")

def ask_yes_no(prompt: str) -> bool:
    """Ask a yes/no question (y/n)."""
    while True:
        raw = input(prompt).strip().lower()
        if raw in ("y", "yes"):
            return True
        if raw in ("n", "no"):
            return False
        print("Please answer with 'y' or 'n'.")

# -----------------------------
# One Run (for a chosen dataset)
# -----------------------------

def do_one_run(choice: int):
    """Execute the mining pipeline for the selected dataset number."""
    path = find_existing_path(CANDIDATE_FILES[choice])
    if not path:
        print("\n[!] Could not find any of the expected files for your choice:")
        for candidate in CANDIDATE_FILES[choice]:
            print(f"    - {candidate}")
        print("    Place the CSV next to this .py file and try again.")
        return

    print(f"\n[+] Loading: {path.name}")
    df = read_csv_safely(path)

    try:
        transactions = extract_transactions(df)
    except Exception as e:
        print(f"\n[!] Failed to parse transactions: {e}")
        return

    n_tx = len(transactions)
    unique_items = sorted({itm for tx in transactions for itm in tx})
    print(f"[i] Parsed {n_tx} transactions with {len(unique_items)} unique items.")

    min_support = ask_float("\nInput your minimum support value (0-1), e.g. 0.02: ", 0.0, 1.0)
    min_confidence = ask_float("Input your minimum confidence value (0-1), e.g. 0.4: ", 0.0, 1.0)

    # ---- Brute-force baseline ----
    t0 = time.time()
    frequent_patterns, pattern_counts = brute_force_apriori(transactions, min_support)
    rules_bf = generate_rules_from_patterns(frequent_patterns, pattern_counts, n_tx, min_confidence)
    rt_bf = time.time() - t0

    print_frequent_patterns(frequent_patterns, pattern_counts, n_tx)
    print_rules(rules_bf)
    print(f"\n[Runtime] Brute-force Apriori: {rt_bf:.4f} s")

    # ---- mlxtend pipelines ----
    df_enc = encode_transactions(transactions)

    fi_ap, rules_ap, rt_ap = run_mlxtend_apriori(df_enc, min_support, min_confidence)
    print_mlxtend_outputs("Apriori (mlxtend)", fi_ap, rules_ap)
    print(f"\n[Runtime] mlxtend Apriori: {rt_ap:.4f} s")

    fi_fp, rules_fp, rt_fp = run_mlxtend_fpgrowth(df_enc, min_support, min_confidence)
    print_mlxtend_outputs("FP-Growth (mlxtend)", fi_fp, rules_fp)
    print(f"\n[Runtime] mlxtend FP-Growth: {rt_fp:.4f} s")

    # ---- Timing Summary (no speed column) ----
    bf_itemsets = len(frequent_patterns)
    bf_rules = len(rules_bf)
    ap_itemsets = 0 if fi_ap is None or fi_ap.empty else int(fi_ap.shape[0])
    ap_rules = 0 if rules_ap is None or rules_ap.empty else int(rules_ap.shape[0])
    fp_itemsets = 0 if fi_fp is None or fi_fp.empty else int(fi_fp.shape[0])
    fp_rules = 0 if rules_fp is None or rules_fp.empty else int(rules_fp.shape[0])

    print_header("Timing Summary")
    print(f"Dataset: {path.name} | Transactions: {n_tx} | Unique items: {len(unique_items)}\n")
    print(f"{'Algorithm':<20} {'Itemsets':>9} {'Rules':>9} {'Runtime (s)':>14}")
    print("-" * 56)
    print(f"{'Brute-force Apriori':<20} {bf_itemsets:>9} {bf_rules:>9} {rt_bf:>14.4f}")
    print(f"{'Apriori (mlxtend)':<20} {ap_itemsets:>9} {ap_rules:>9} {rt_ap:>14.4f}")
    print(f"{'FP-Growth (mlxtend)':<20} {fp_itemsets:>9} {fp_rules:>9} {rt_fp:>14.4f}")

    print_header("Done")

# -----------------------------
# Main Loop
# -----------------------------

def main():
    print_header("Data Mining Midterm Project")
    try:
        while True:
            choice = ask_menu_choice()
            if choice == 0:   # Exit
                print("Goodbye!")
                return
            do_one_run(choice)
            print()  # spacing
            if not ask_yes_no("Run again? (y/n): "):
                print("Goodbye!")
                return
            print()  # spacing
    except KeyboardInterrupt:
        print("\nInterrupted. Goodbye!")

if __name__ == "__main__":
    main()
