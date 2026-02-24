"""
GNN-based property predictor for polymer electrolytes.

Ported from prompt-optimization-work-jan-8 with:
- Clean SurrogatePredictor interface
- @register decorators for all supported properties
- Graceful handling of missing model files
"""
from __future__ import annotations

import functools
import os
from typing import List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Linear, ReLU, Sequential
from torch.utils.data import Dataset
from rdkit import Chem
from rdkit.Chem import AllChem

from .base import SurrogatePredictor
from .registry import register

# ──────────────────────────────────────────────────────────────────────────────
# Feature encodings
# ──────────────────────────────────────────────────────────────────────────────

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


def _onehot(feature_list, cur_feature):
    assert cur_feature in feature_list, f"{cur_feature} not in {feature_list}"
    vector = [0] * len(feature_list)
    vector[feature_list.index(cur_feature)] = 1
    return vector


def _process_smiles(smiles: str, form_ring: bool = True, has_H: bool = False):
    mol = Chem.MolFromSmiles(smiles)
    if has_H and mol:
        mol = Chem.AddHs(mol)
    if form_ring and mol:
        rxn = AllChem.ReactionFromSmarts("([Cu][*:1].[*:2][Au])>>[*:1]-[*:2]")
        results = rxn.RunReactants([mol])
        if not (len(results) == 1 and len(results[0]) == 1):
            rxn = AllChem.ReactionFromSmarts("([Cu]=[*:1].[*:2]=[Au])>>[*:1]=[*:2]")
            results = rxn.RunReactants([mol])
        if len(results) == 1 and len(results[0]) == 1:
            mol = results[0][0]
    if mol:
        try:
            Chem.SanitizeMol(mol)
        except Exception:
            return None
    return mol


# ──────────────────────────────────────────────────────────────────────────────
# GNN architecture
# ──────────────────────────────────────────────────────────────────────────────

class PolymerDataset(Dataset):
    def __init__(self, smiles_list: List[str], log10: bool = True,
                 form_ring: bool = True, has_H: bool = False):
        self.raw_data = [("test", s, 1.0) for s in smiles_list]
        self.log10 = log10
        self.form_ring = form_ring
        self.has_H = has_H

    def __len__(self):
        return len(self.raw_data)

    @functools.lru_cache(maxsize=None)
    def __getitem__(self, idx):
        from torch_geometric.data import Data
        _, smiles, target = self.raw_data[idx]
        mol = _process_smiles(smiles, self.form_ring, self.has_H)
        if mol is None:
            return None

        target_val = float(target)
        if self.log10:
            target_val = np.log10(max(target_val, 1e-30))
        target_t = torch.tensor(target_val).float()

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

        edge_index = torch.tensor(edge_indices).t().to(torch.long).view(2, -1)
        edge_attr = torch.tensor(edge_attrs, dtype=torch.float)

        if edge_index.numel() > 0:
            perm = (edge_index[0] * x_t.size(0) + edge_index[1]).argsort()
            edge_index, edge_attr = edge_index[:, perm], edge_attr[perm]

        return Data(x=x_t, edge_index=edge_index, edge_attr=edge_attr,
                    y=target_t, smiles=smiles)


class PolymerNet(torch.nn.Module):
    def __init__(self, node_in_len, edge_in_len, fea_len=16, n_layers=4, n_h=2):
        super().__init__()
        from torch_geometric.nn import CGConv, GlobalAttention
        self.node_embed = Linear(node_in_len, fea_len)
        self.edge_embed = Linear(edge_in_len, fea_len)
        self.cgconvs = nn.ModuleList([
            CGConv(fea_len, fea_len, aggr="mean", batch_norm=True)
            for _ in range(n_layers)
        ])
        self.pool = GlobalAttention(
            gate_nn=Sequential(Linear(fea_len, fea_len), Linear(fea_len, 1)),
            nn=Sequential(Linear(fea_len, fea_len), Linear(fea_len, fea_len)),
        )
        self.hs = nn.ModuleList([Linear(fea_len, fea_len) for _ in range(n_h - 1)])
        self.out = Linear(fea_len, 1)

    def forward(self, data):
        out = F.leaky_relu(self.node_embed(data.x))
        ea = F.leaky_relu(self.edge_embed(data.edge_attr))
        for conv in self.cgconvs:
            out = conv(out, data.edge_index, ea)
        out = self.pool(out, data.batch)
        for h in self.hs:
            out = F.leaky_relu(h(out))
        return torch.squeeze(self.out(out), dim=-1)


# ──────────────────────────────────────────────────────────────────────────────
# GNN property configs
# ──────────────────────────────────────────────────────────────────────────────

_GNN_PROPS = {
    "conductivity": {
        "mean": -4.262819, "std": 0.222358, "log10": True,
        "scaling": 1000.0,  # S/cm → mS/cm
        "property_name": "Conductivity", "property_units": "mS/cm", "maximize": True,
    },
    "li_diffusivity": {
        "mean": -7.81389, "std": 0.205920, "log10": True, "scaling": 1.0,
        "property_name": "Li Diffusivity", "property_units": "cm²/s", "maximize": True,
    },
    "poly_diffusivity": {
        "mean": -7.841585, "std": 0.256285, "log10": True, "scaling": 1.0,
        "property_name": "Polymer Diffusivity", "property_units": "cm²/s", "maximize": True,
    },
    "tfsi_diffusivity": {
        "mean": -7.60879, "std": 0.217374, "log10": True, "scaling": 1.0,
        "property_name": "TFSI Diffusivity", "property_units": "cm²/s", "maximize": True,
    },
    "transference_number": {
        "mean": 0.0623139, "std": 0.281334, "log10": False, "scaling": 1.0,
        "property_name": "Transference Number", "property_units": "", "maximize": True,
    },
}


# ──────────────────────────────────────────────────────────────────────────────
# GNNPredictor class
# ──────────────────────────────────────────────────────────────────────────────

class GNNPredictor(SurrogatePredictor):
    """
    GNN-based surrogate predictor for polymer electrolyte properties.
    Model weights are loaded from `model_base_path/pre_trained_gnns/<prop>.pth`.
    """

    def __init__(self, prop_key: str, model_base_path: str):
        if prop_key not in _GNN_PROPS:
            raise ValueError(f"Unknown GNN property '{prop_key}'. Available: {list(_GNN_PROPS)}")
        cfg = _GNN_PROPS[prop_key]
        self._prop_key = prop_key
        self._cfg = cfg
        self._model_base_path = model_base_path
        # Populate SurrogatePredictor attributes
        self.property_name = cfg["property_name"]
        self.property_units = cfg["property_units"]
        self.maximize = cfg["maximize"]

    @property
    def _model_path(self) -> str:
        return os.path.join(self._model_base_path, "pre_trained_gnns", f"{self._prop_key}.pth")

    def predict(self, smiles_list: List[str]) -> List[Optional[float]]:
        if not smiles_list:
            return []

        cfg = self._cfg
        log10 = cfg["log10"]
        mean, std = cfg["mean"], cfg["std"]
        scaling = cfg["scaling"]

        dataset = PolymerDataset(smiles_list, log10=log10, form_ring=True, has_H=False)

        valid_indices, valid_data = [], []
        for i in range(len(dataset)):
            d = dataset[i]
            if d is not None:
                valid_indices.append(i)
                valid_data.append(d)

        if not valid_data:
            return [None] * len(smiles_list)

        if not os.path.exists(self._model_path):
            print(f"[GNNPredictor] Model file not found: {self._model_path}")
            return [None] * len(smiles_list)

        from torch_geometric.data import DataLoader
        loader = DataLoader(valid_data, batch_size=128, shuffle=False)
        example = valid_data[0]
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        model = PolymerNet(
            example.num_features, example.num_edge_features,
            fea_len=16, n_layers=4, n_h=2,
        ).to(device)

        try:
            model.load_state_dict(torch.load(self._model_path, map_location=device))
        except Exception as e:
            print(f"[GNNPredictor] Error loading model: {e}")
            return [None] * len(smiles_list)

        model.eval()
        preds_map = {}
        pointer = 0
        with torch.no_grad():
            for batch in loader:
                batch = batch.to(device)
                pred = model(batch)
                pred = pred * std + mean
                pred_np = pred.cpu().numpy()
                for k in range(len(pred_np)):
                    val = float(pred_np[k])
                    if log10:
                        val = 10 ** val
                    val *= scaling
                    preds_map[valid_indices[pointer]] = val
                    pointer += 1

        return [preds_map.get(i) for i in range(len(smiles_list))]


# ──────────────────────────────────────────────────────────────────────────────
# Register all GNN property surrogates
# ──────────────────────────────────────────────────────────────────────────────

def _make_gnn_class(prop_key: str) -> type:
    """Dynamically create a GNNPredictor subclass bound to a specific property."""
    class _GNNProp(GNNPredictor):
        def __init__(self, model_base_path: str):
            super().__init__(prop_key, model_base_path)
    _GNNProp.__name__ = f"GNN_{prop_key}"
    return _GNNProp


for _prop_key in _GNN_PROPS:
    _cls = _make_gnn_class(_prop_key)
    register(f"gnn_{_prop_key}")(_cls)
