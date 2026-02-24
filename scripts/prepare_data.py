"""
Prepare a small stratified subset of the conductivity dataset for fast experimentation.
Uses 100 molecules total → ~80 train / 20 test, stratified by log-conductivity deciles.
"""
import numpy as np
import pandas as pd
from pathlib import Path

RAW_CSV = "/noether/s0/dxb5775/prompt-optimization-work-jan-8/data/PolyGen-train-set-from-HTP-MD.csv"
OUT_DIR = Path("/noether/s0/dxb5775/agentic-prompt-optimization/data")
TOTAL_SUBSET = 100     # total molecules to work with
TRAIN_FRAC   = 0.80
SEED         = 42

def main():
    OUT_DIR.mkdir(exist_ok=True)
    df = pd.read_csv(RAW_CSV)[["mol_smiles", "conductivity"]].dropna()
    df = df[df["conductivity"] > 0].reset_index(drop=True)
    df["log_cond"] = np.log10(df["conductivity"])

    # Stratified sample of TOTAL_SUBSET from the full 6024
    df["stratum"] = pd.qcut(df["log_cond"], q=10, labels=False, duplicates="drop")
    rng = np.random.default_rng(SEED)
    subset_idx = []
    per_stratum = max(1, TOTAL_SUBSET // df["stratum"].nunique())
    for _, grp in df.groupby("stratum"):
        n = min(len(grp), per_stratum)
        subset_idx.extend(rng.choice(grp.index, size=n, replace=False).tolist())

    # Trim or pad to exactly TOTAL_SUBSET
    rng.shuffle(subset_idx)
    subset_idx = subset_idx[:TOTAL_SUBSET]
    subset = df.loc[subset_idx, ["mol_smiles", "conductivity"]].reset_index(drop=True)

    # Train / test split, stratified by same strata
    subset["stratum"] = pd.qcut(subset["log_cond"] if "log_cond" in subset.columns
                                else np.log10(subset["conductivity"]),
                                q=5, labels=False, duplicates="drop")
    train_idx, test_idx = [], []
    for _, grp in subset.groupby("stratum"):
        idx = grp.index.tolist()
        rng.shuffle(idx)
        n_train = max(1, int(len(idx) * TRAIN_FRAC))
        train_idx.extend(idx[:n_train])
        test_idx.extend(idx[n_train:])

    train = subset.loc[train_idx, ["mol_smiles", "conductivity"]].reset_index(drop=True)
    test  = subset.loc[test_idx,  ["mol_smiles", "conductivity"]].reset_index(drop=True)

    train.to_csv(OUT_DIR / "train.csv", index=False)
    test.to_csv(OUT_DIR  / "test.csv",  index=False)

    print(f"Total subset: {len(subset)}  →  Train: {len(train)}  Test: {len(test)}")
    print(f"\nTrain conductivity (log10 mS/cm):")
    print(np.log10(train["conductivity"]).describe().round(3))
    print(f"\nTest conductivity (log10 mS/cm):")
    print(np.log10(test["conductivity"]).describe().round(3))
    print(f"\nSaved to {OUT_DIR}/")

if __name__ == "__main__":
    main()
