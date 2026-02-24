"""
Train GNN surrogate models for glass transition temperature (Tg).

Three architectures tested:
  A. GCN (small, 2 layers)          — baseline
  B. CGConv + GlobalAttention (med) — same family as conductivity GNN
  C. CGConv + GlobalAttention (lg)  — deeper variant

SMILES use '*' as repeat-unit stub (not [Cu]/[Au]).
We replace * with [Ti] so RDKit parses it, then ring-close with Ti-Ti → bond.

Output:
  models/tg/best_model.pth     — best weights by val RMSE
  models/tg/model_config.json  — architecture hyper-params
  models/tg/scaler.json        — mean/std for Tg normalisation
  models/tg/training_report.txt
"""
import json
import math
import os
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Linear, Sequential
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
OUT_DIR = ROOT / "models" / "tg"
OUT_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

# ── Feature maps (same as conductivity GNN) ──────────────────────────────────
X_MAP = {
    "atomic_num": list(range(0, 119)),
    "degree": list(range(0, 11)),
    "formal_charge": list(range(-5, 7)),
    "num_hs": list(range(0, 9)),
    "hybridization": ["UNSPECIFIED", "S", "SP", "SP2", "SP3", "SP3D", "SP3D2", "OTHER"],
    "is_aromatic": [False, True],
    "is_in_ring": [False, True],
}
E_MAP = {
    "bond_type": ["misc", "SINGLE", "DOUBLE", "TRIPLE", "AROMATIC"],
    "stereo": ["STEREONONE", "STEREOZ", "STEREOE", "STEREOCIS", "STEREOTRANS", "STEREOANY"],
    "is_conjugated": [False, True],
}
NODE_IN = sum(len(v) for v in X_MAP.values())   # 156
EDGE_IN = sum(len(v) for v in E_MAP.values())   # 13


def _onehot(feature_list, cur_feature):
    if cur_feature not in feature_list:
        return [0] * len(feature_list)
    v = [0] * len(feature_list)
    v[feature_list.index(cur_feature)] = 1
    return v


def smiles_to_data(smiles: str, target: float):
    """Convert polymer SMILES (with *) to torch_geometric Data. Returns None if invalid."""
    from rdkit import Chem
    from rdkit.Chem import AllChem
    from torch_geometric.data import Data

    # Replace * with [Ti] so RDKit parses it, then ring-close Ti-Ti
    smi = smiles.replace("*", "[Ti]")
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return None
    # Ring-close: [Ti]-X ... Y-[Ti] → X-Y
    rxn = AllChem.ReactionFromSmarts("([Ti][*:1].[*:2][Ti])>>[*:1]-[*:2]")
    res = rxn.RunReactants([mol])
    if res and len(res[0]) == 1:
        mol = res[0][0]
    try:
        Chem.SanitizeMol(mol)
    except Exception:
        return None

    xs = []
    for atom in mol.GetAtoms():
        x = (
            _onehot(X_MAP["atomic_num"], atom.GetAtomicNum())
            + _onehot(X_MAP["degree"], atom.GetTotalDegree())
            + _onehot(X_MAP["formal_charge"], atom.GetFormalCharge())
            + _onehot(X_MAP["num_hs"], atom.GetTotalNumHs())
            + _onehot(X_MAP["hybridization"], str(atom.GetHybridization()))
            + _onehot(X_MAP["is_aromatic"], atom.GetIsAromatic())
            + _onehot(X_MAP["is_in_ring"], atom.IsInRing())
        )
        xs.append(x)
    x_t = torch.tensor(xs, dtype=torch.float)

    edge_indices, edge_attrs = [], []
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        e = (
            _onehot(E_MAP["bond_type"], str(bond.GetBondType()))
            + _onehot(E_MAP["stereo"], str(bond.GetStereo()))
            + _onehot(E_MAP["is_conjugated"], bond.GetIsConjugated())
        )
        edge_indices += [[i, j], [j, i]]
        edge_attrs += [e, e]

    if not edge_indices:
        return None
    edge_index = torch.tensor(edge_indices).t().long().view(2, -1)
    edge_attr = torch.tensor(edge_attrs, dtype=torch.float)
    perm = (edge_index[0] * x_t.size(0) + edge_index[1]).argsort()
    edge_index, edge_attr = edge_index[:, perm], edge_attr[perm]

    return Data(x=x_t, edge_index=edge_index, edge_attr=edge_attr,
                y=torch.tensor(target, dtype=torch.float))


# ── Architecture A: simple GCN ────────────────────────────────────────────────
class GCNNet(torch.nn.Module):
    """Lightweight GCN baseline (no edge features)."""
    def __init__(self, node_in=NODE_IN, hidden=64, n_layers=3, dropout=0.1):
        super().__init__()
        from torch_geometric.nn import GCNConv, global_mean_pool
        self.convs = nn.ModuleList()
        self.convs.append(GCNConv(node_in, hidden))
        for _ in range(n_layers - 1):
            self.convs.append(GCNConv(hidden, hidden))
        self.pool = global_mean_pool
        self.fc1 = Linear(hidden, hidden)
        self.fc2 = Linear(hidden, 1)
        self.drop = nn.Dropout(dropout)

    def forward(self, data):
        x, ei, batch = data.x, data.edge_index, data.batch
        for conv in self.convs:
            x = F.leaky_relu(conv(x, ei))
            x = self.drop(x)
        x = self.pool(x, batch)
        x = F.leaky_relu(self.fc1(x))
        return self.fc2(x).squeeze(-1)


# ── Architecture B: CGConv medium ────────────────────────────────────────────
class CGConvNet(torch.nn.Module):
    """CGConv + GlobalAttention (same family as conductivity GNN)."""
    def __init__(self, node_in=NODE_IN, edge_in=EDGE_IN, fea_len=64,
                 n_layers=4, n_h=3, dropout=0.1):
        super().__init__()
        from torch_geometric.nn import CGConv, GlobalAttention
        self.node_embed = Linear(node_in, fea_len)
        self.edge_embed = Linear(edge_in, fea_len)
        self.convs = nn.ModuleList([
            CGConv(fea_len, fea_len, aggr="add", batch_norm=True)
            for _ in range(n_layers)
        ])
        self.pool = GlobalAttention(
            gate_nn=Sequential(Linear(fea_len, fea_len), nn.ReLU(), Linear(fea_len, 1)),
            nn=Sequential(Linear(fea_len, fea_len), nn.ReLU(), Linear(fea_len, fea_len)),
        )
        self.heads = nn.ModuleList([Linear(fea_len, fea_len) for _ in range(n_h - 1)])
        self.out = Linear(fea_len, 1)
        self.drop = nn.Dropout(dropout)

    def forward(self, data):
        x = F.leaky_relu(self.node_embed(data.x))
        ea = F.leaky_relu(self.edge_embed(data.edge_attr))
        for conv in self.convs:
            x = conv(x, data.edge_index, ea)
            x = self.drop(x)
        x = self.pool(x, data.batch)
        for h in self.heads:
            x = F.leaky_relu(h(x))
        return self.out(x).squeeze(-1)


# ── Architecture C: CGConv large ─────────────────────────────────────────────
class CGConvNetLarge(CGConvNet):
    def __init__(self):
        super().__init__(fea_len=128, n_layers=6, n_h=4, dropout=0.15)


# ── Dataset helpers ───────────────────────────────────────────────────────────
def build_dataset(df, mean_tg, std_tg):
    data_list = []
    for _, row in df.iterrows():
        normalized = (row["tg"] - mean_tg) / std_tg
        d = smiles_to_data(row["SMILES"], normalized)
        if d is not None:
            data_list.append(d)
    return data_list


def make_loader(data_list, batch_size=64, shuffle=True):
    from torch_geometric.loader import DataLoader
    return DataLoader(data_list, batch_size=batch_size, shuffle=shuffle)


# ── Training loop ─────────────────────────────────────────────────────────────
def train_model(model, train_loader, val_loader, epochs=80, lr=1e-3, patience=15):
    model = model.to(DEVICE)
    opt = Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    sched = CosineAnnealingLR(opt, T_max=epochs, eta_min=1e-5)
    best_val = float("inf")
    best_state = None
    no_improve = 0
    history = []

    for ep in range(1, epochs + 1):
        model.train()
        total_loss = 0
        for batch in train_loader:
            batch = batch.to(DEVICE)
            opt.zero_grad()
            pred = model(batch)
            loss = F.mse_loss(pred, batch.y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total_loss += loss.item() * batch.num_graphs
        sched.step()

        val_rmse = evaluate(model, val_loader)
        train_rmse = math.sqrt(total_loss / sum(b.num_graphs for b in train_loader))
        history.append({"epoch": ep, "train_rmse": train_rmse, "val_rmse": val_rmse})

        if ep % 10 == 0 or ep == 1:
            print(f"  Epoch {ep:3d} | train RMSE {train_rmse:.4f} | val RMSE {val_rmse:.4f}")

        if val_rmse < best_val:
            best_val = val_rmse
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"  Early stop at epoch {ep} (no improvement for {patience} epochs)")
                break

    return best_val, best_state, history


def evaluate(model, loader):
    model.eval()
    preds, targets = [], []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(DEVICE)
            pred = model(batch)
            preds.append(pred.cpu())
            targets.append(batch.y.cpu())
    preds = torch.cat(preds).numpy()
    targets = torch.cat(targets).numpy()
    return float(np.sqrt(np.mean((preds - targets) ** 2)))


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    print(f"[TgGNN] Device: {DEVICE}")
    df = pd.read_csv("/noether/s0/dxb5775/tg_raw.csv")
    df = df.dropna().reset_index(drop=True)
    print(f"[TgGNN] Loaded {len(df)} molecules. Tg range: [{df.tg.min():.1f}, {df.tg.max():.1f}] °C")

    # Train/val/test split (70/15/15)
    idx = list(range(len(df)))
    random.shuffle(idx)
    n = len(idx)
    n_train = int(n * 0.70)
    n_val   = int(n * 0.15)
    train_df = df.iloc[idx[:n_train]]
    val_df   = df.iloc[idx[n_train:n_train + n_val]]
    test_df  = df.iloc[idx[n_train + n_val:]]

    mean_tg = float(train_df["tg"].mean())
    std_tg  = float(train_df["tg"].std())
    print(f"[TgGNN] Train/Val/Test: {len(train_df)}/{len(val_df)}/{len(test_df)}")
    print(f"[TgGNN] Normalisation: mean={mean_tg:.2f} std={std_tg:.2f}")

    print("[TgGNN] Building graph datasets...")
    train_data = build_dataset(train_df, mean_tg, std_tg)
    val_data   = build_dataset(val_df, mean_tg, std_tg)
    test_data  = build_dataset(test_df, mean_tg, std_tg)
    print(f"[TgGNN] Valid graphs: train={len(train_data)} val={len(val_data)} test={len(test_data)}")

    train_ld = make_loader(train_data, batch_size=64, shuffle=True)
    val_ld   = make_loader(val_data, batch_size=128, shuffle=False)
    test_ld  = make_loader(test_data, batch_size=128, shuffle=False)

    architectures = [
        ("GCN_small",   GCNNet(hidden=64, n_layers=3),                  70,  1e-3),
        ("CGConv_med",  CGConvNet(fea_len=64, n_layers=4, n_h=3),        90,  8e-4),
        ("CGConv_large",CGConvNetLarge(),                                100,  5e-4),
    ]

    results = []
    report_lines = [f"Tg GNN Training Report", f"{'='*60}",
                    f"Train: {len(train_data)}  Val: {len(val_data)}  Test: {len(test_data)}",
                    f"Tg mean: {mean_tg:.2f}  std: {std_tg:.2f}", ""]

    for name, model, epochs, lr in architectures:
        n_params = sum(p.numel() for p in model.parameters())
        print(f"\n[TgGNN] ── Architecture: {name} ({n_params:,} params) ──")
        report_lines.append(f"Architecture: {name}  ({n_params:,} params)")
        val_rmse, best_state, history = train_model(
            model, train_ld, val_ld, epochs=epochs, lr=lr, patience=15
        )
        # Test RMSE with best weights
        model.load_state_dict(best_state)
        test_rmse_norm = evaluate(model, test_ld)
        # Convert back to °C
        val_rmse_c = val_rmse * std_tg
        test_rmse_c = test_rmse_norm * std_tg
        print(f"  Best val RMSE: {val_rmse_c:.2f} °C  |  Test RMSE: {test_rmse_c:.2f} °C")
        report_lines.append(f"  Val RMSE: {val_rmse_c:.2f} °C  Test RMSE: {test_rmse_c:.2f} °C\n")
        results.append({
            "name": name, "model": model, "state": best_state,
            "val_rmse": val_rmse, "val_rmse_c": val_rmse_c,
            "test_rmse_c": test_rmse_c, "history": history,
        })

    best = min(results, key=lambda r: r["val_rmse"])
    print(f"\n[TgGNN] ✓ Best architecture: {best['name']} (val RMSE {best['val_rmse_c']:.2f} °C)")
    report_lines.append(f"WINNER: {best['name']}  (val RMSE {best['val_rmse_c']:.2f} °C, test RMSE {best['test_rmse_c']:.2f} °C)")

    # Save model
    torch.save(best["state"], OUT_DIR / "best_model.pth")
    with open(OUT_DIR / "scaler.json", "w") as f:
        json.dump({"mean": mean_tg, "std": std_tg, "architecture": best["name"]}, f, indent=2)
    # Save detailed architecture config
    cfg = {
        "architecture": best["name"],
        "val_rmse_c": best["val_rmse_c"],
        "test_rmse_c": best["test_rmse_c"],
        "node_in": NODE_IN,
        "edge_in": EDGE_IN,
        "mean_tg": mean_tg,
        "std_tg": std_tg,
    }
    if best["name"] == "GCN_small":
        cfg.update({"hidden": 64, "n_layers": 3})
    elif best["name"] == "CGConv_med":
        cfg.update({"fea_len": 64, "n_layers": 4, "n_h": 3})
    else:
        cfg.update({"fea_len": 128, "n_layers": 6, "n_h": 4})
    with open(OUT_DIR / "model_config.json", "w") as f:
        json.dump(cfg, f, indent=2)

    # Training history
    with open(OUT_DIR / "training_report.txt", "w") as f:
        f.write("\n".join(report_lines) + "\n\nTraining History:\n")
        for r in results:
            f.write(f"\n{r['name']}:\n")
            for h in r["history"]:
                f.write(f"  ep={h['epoch']:3d}  train={h['train_rmse']:.4f}  val={h['val_rmse']:.4f}\n")

    print(f"\n[TgGNN] Saved model to {OUT_DIR}/best_model.pth")
    print(f"[TgGNN] Config:  {OUT_DIR}/model_config.json")
    print(f"[TgGNN] Report:  {OUT_DIR}/training_report.txt")

    # Quick inference sanity check
    from apo.surrogates.tg_predictor import TgGNNPredictor
    pred = TgGNNPredictor(model_path=str(OUT_DIR / "best_model.pth"),
                          scaler_path=str(OUT_DIR / "scaler.json"))
    sample = test_df["SMILES"].iloc[:3].tolist()
    actual = test_df["tg"].iloc[:3].tolist()
    preds = pred.predict(sample)
    print("\n[TgGNN] Sanity check:")
    for s, a, p in zip(sample, actual, preds):
        pred_str = f"{p:.1f}" if p is not None else "None"
        print(f"  {s[:40]:40s}  actual={a:.1f}  pred={pred_str}")

if __name__ == "__main__":
    main()
