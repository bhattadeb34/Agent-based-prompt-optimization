"""
Tg (glass transition temperature) GNN surrogate predictor.

Registered as 'gnn_tg' in the APO surrogate registry.
Loads model from models/tg/ (created by scripts/train_tg_gnn.py).

SMILES format: use '*' as repeat-unit placeholder (PolyInfo convention).
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Linear, Sequential

from .base import SurrogatePredictor
from .registry import register

# ── Feature maps shared with gnn_predictor ────────────────────────────────────
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
NODE_IN = sum(len(v) for v in X_MAP.values())
EDGE_IN = sum(len(v) for v in E_MAP.values())

# Default model path (relative to this file's package root)
_PKG_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_MODEL_PATH  = _PKG_ROOT / "models" / "tg" / "best_model.pth"
_DEFAULT_SCALER_PATH = _PKG_ROOT / "models" / "tg" / "scaler.json"
_DEFAULT_CFG_PATH    = _PKG_ROOT / "models" / "tg" / "model_config.json"


def _onehot(feature_list, cur_feature):
    if cur_feature not in feature_list:
        return [0] * len(feature_list)
    v = [0] * len(feature_list)
    v[feature_list.index(cur_feature)] = 1
    return v


def smiles_to_data(smiles: str):
    """Convert polymer SMILES (with *) to torch_geometric Data. Returns None if invalid."""
    from rdkit import Chem
    from rdkit.Chem import AllChem
    from torch_geometric.data import Data

    smi = smiles.replace("*", "[Ti]")
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return None
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
    edge_attr  = torch.tensor(edge_attrs, dtype=torch.float)
    perm = (edge_index[0] * x_t.size(0) + edge_index[1]).argsort()
    edge_index, edge_attr = edge_index[:, perm], edge_attr[perm]
    return Data(x=x_t, edge_index=edge_index, edge_attr=edge_attr,
                y=torch.tensor(0.0))  # dummy target for inference


# ── Model architectures (mirror of train_tg_gnn.py) ─────────────────────────

class GCNNet(torch.nn.Module):
    def __init__(self, node_in=NODE_IN, hidden=64, n_layers=3, dropout=0.1):
        super().__init__()
        from torch_geometric.nn import GCNConv
        self.convs = nn.ModuleList()
        self.convs.append(GCNConv(node_in, hidden))
        for _ in range(n_layers - 1):
            self.convs.append(GCNConv(hidden, hidden))
        self.fc1 = Linear(hidden, hidden)
        self.fc2 = Linear(hidden, 1)
        self.drop = nn.Dropout(dropout)

    def forward(self, data):
        from torch_geometric.nn import global_mean_pool
        x, ei = data.x, data.edge_index
        for conv in self.convs:
            x = F.leaky_relu(conv(x, ei))
            x = self.drop(x)
        x = global_mean_pool(x, data.batch)
        x = F.leaky_relu(self.fc1(x))
        return self.fc2(x).squeeze(-1)


class CGConvNet(torch.nn.Module):
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


class CGConvNetLarge(CGConvNet):
    def __init__(self):
        super().__init__(fea_len=128, n_layers=6, n_h=4, dropout=0.15)


def _build_model_from_config(cfg: dict) -> torch.nn.Module:
    arch = cfg.get("architecture", "CGConv_med")
    if arch == "GCN_small":
        return GCNNet(hidden=cfg.get("hidden", 64), n_layers=cfg.get("n_layers", 3))
    elif arch == "CGConv_large":
        return CGConvNetLarge()
    else:  # CGConv_med or unknown
        return CGConvNet(
            fea_len=cfg.get("fea_len", 64),
            n_layers=cfg.get("n_layers", 4),
            n_h=cfg.get("n_h", 3),
        )


# ── TgGNNPredictor ────────────────────────────────────────────────────────────

class TgGNNPredictor(SurrogatePredictor):
    """GNN surrogate for glass transition temperature (Tg) in °C."""

    property_name = "Glass Transition Temperature"
    property_units = "°C"
    maximize = True  # higher Tg = more thermally stable

    def __init__(
        self,
        model_path: str = str(_DEFAULT_MODEL_PATH),
        scaler_path: str = str(_DEFAULT_SCALER_PATH),
        config_path: str = str(_DEFAULT_CFG_PATH),
    ):
        self._model_path = model_path
        self._scaler_path = scaler_path
        self._config_path = config_path
        self._model = None
        self._mean = None
        self._std = None

    def _load(self):
        """Lazy-load model and scaler."""
        if self._model is not None:
            return

        if not os.path.exists(self._scaler_path):
            raise FileNotFoundError(f"Tg scaler not found: {self._scaler_path}. "
                                    f"Run scripts/train_tg_gnn.py first.")
        with open(self._scaler_path) as f:
            scaler = json.load(f)
        self._mean = scaler["mean"]
        self._std  = scaler["std"]

        cfg = {}
        if os.path.exists(self._config_path):
            with open(self._config_path) as f:
                cfg = json.load(f)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = _build_model_from_config(cfg).to(device)

        if not os.path.exists(self._model_path):
            raise FileNotFoundError(f"Tg model weights not found: {self._model_path}. "
                                    f"Run scripts/train_tg_gnn.py first.")
        model.load_state_dict(
            torch.load(self._model_path, map_location=device, weights_only=False)
        )
        model.eval()
        self._model = model
        self._device = device
        print(f"[TgGNNPredictor] Loaded {cfg.get('architecture','?')} "
              f"(val RMSE {cfg.get('val_rmse_c', '?'):.1f} °C) from {self._model_path}")

    def predict(self, smiles_list: List[str]) -> List[Optional[float]]:
        if not smiles_list:
            return []
        self._load()

        from torch_geometric.loader import DataLoader
        valid_idx, valid_data = [], []
        for i, smi in enumerate(smiles_list):
            d = smiles_to_data(smi)
            if d is not None:
                valid_idx.append(i)
                valid_data.append(d)

        if not valid_data:
            return [None] * len(smiles_list)

        loader = DataLoader(valid_data, batch_size=128, shuffle=False)
        preds_map = {}
        ptr = 0
        with torch.no_grad():
            for batch in loader:
                batch = batch.to(self._device)
                pred = self._model(batch).cpu().numpy()
                for k in range(len(pred)):
                    tg = float(pred[k]) * self._std + self._mean
                    preds_map[valid_idx[ptr]] = tg
                    ptr += 1

        return [preds_map.get(i) for i in range(len(smiles_list))]


# ── Register with APO surrogate registry ─────────────────────────────────────
@register("gnn_tg")
class _TgPredictor(TgGNNPredictor):
    """Registry-compatible Tg predictor. model_base_path ignored (uses models/tg/ by default)."""
    def __init__(self, model_base_path: str = ""):
        super().__init__()
