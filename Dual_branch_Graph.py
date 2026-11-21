"""Dual-branch PV forecasting with optional station graph and solar elevation.

The power branch embeds historical PV power and can aggregate station tokens
over a graph. The Meteorology branch embeds context features and time marks.
Cross-attention combines the branches before the forecast head.

MultiPower accepts [B, L, stations, features] or flattened [B, L, channels]
inputs. In the former, graph_power_indices selects stations and
power_feature_index selects the PV feature. In the latter, graph_power_indices
selects PV channels. Adjacency can come from a .npy matrix with a station list,
a top-k similarity text file, or a tensor passed to forward.
"""

from __future__ import annotations

import os
import re
import math
import warnings
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn


# -----------------------------
# Utility functions
# -----------------------------

def _as_bool(x) -> bool:
    if isinstance(x, bool):
        return x
    if isinstance(x, str):
        return x.lower() in {"1", "true", "yes", "y", "on"}
    return bool(x)


def _parse_indices(value, default: Optional[List[int]] = None, modulo: Optional[int] = None) -> List[int]:
    """Parse int/list/tuple/comma-separated-string indices and convert negative indices."""
    if value is None:
        return list(default or [])
    if isinstance(value, int):
        values = [value]
    elif isinstance(value, str):
        value = value.strip()
        if not value:
            values = []
        else:
            values = [int(v.strip()) for v in value.split(',') if v.strip()]
    else:
        values = [int(v) for v in value]

    if modulo is not None:
        values = [v % modulo for v in values]
    return values


def _split_data_path(data_path: Optional[str]) -> List[str]:
    """Return station basenames from a data_path string like 'csv/a.csv;csv/b.csv'."""
    if data_path is None:
        return []
    # Support both semicolon and comma separated groups.
    raw = re.split(r"[;,]", str(data_path))
    names = []
    for item in raw:
        item = item.strip()
        if not item:
            continue
        names.append(os.path.basename(item))
    return names


def _maybe_join(root_path: Optional[str], path: Optional[str]) -> Optional[str]:
    if path is None or str(path).strip() == "":
        return None
    path = str(path)
    if os.path.isabs(path):
        return path
    if root_path:
        candidate = os.path.join(str(root_path), path)
        if os.path.exists(candidate):
            return candidate
    return path


def _select_last_dim(x: torch.Tensor, indices: Sequence[int]) -> torch.Tensor:
    """Select indices on the last dimension while supporting negative indices."""
    if len(indices) == 0:
        return x[..., :0]
    dim = x.shape[-1]
    idx = torch.tensor([int(i) % dim for i in indices], dtype=torch.long, device=x.device)
    return torch.index_select(x, dim=-1, index=idx)


def _estimate_solar_elevation_from_time_features(time_features: torch.Tensor, latitude_deg: float = -23.7) -> torch.Tensor:
    """Estimate normalized solar elevation from time embedding features.

    Input shape: [B, L, 5], with columns
        [MinuteOfHour, HourOfDay, DayOfWeek, DayOfMonth, DayOfYear] in [-0.5, 0.5].
    Output shape: [B, L], normalized to [-1, 1].
    """
    if time_features.dim() != 3 or time_features.size(-1) < 5:
        raise ValueError(
            "Solar elevation requires time features with shape [B, L, >=5], "
            "ordered as [MinuteOfHour, HourOfDay, DayOfWeek, DayOfMonth, DayOfYear]."
        )

    minute = (time_features[:, :, 0] + 0.5) * 59.0
    hour = (time_features[:, :, 1] + 0.5) * 23.0
    day_of_year = (time_features[:, :, 4] + 0.5) * 365.0 + 1.0

    latitude = math.radians(float(latitude_deg))
    delta = 23.44 * torch.cos(2 * math.pi * (day_of_year - 172.0) / 365.0)
    delta = torch.deg2rad(delta)

    local_time = hour + minute / 60.0
    hour_angle = torch.deg2rad(15.0 * (local_time - 12.0))

    sin_phi = math.sin(latitude)
    cos_phi = math.cos(latitude)

    elevation_rad = torch.asin(torch.sin(delta) * sin_phi + torch.cos(delta) * cos_phi * torch.cos(hour_angle))
    elevation_deg = torch.rad2deg(elevation_rad).clamp(min=-90.0, max=90.0)
    return elevation_deg / 90.0


# -----------------------------
# Embeddings and heads
# -----------------------------

class PatchEmbedding(nn.Module):
    """
    Simple non-overlapping patch embedding.

    Input:
        x: [B, L, C]
    Output:
        z: [B, P, d_model], where P = L // patch_len
    """
    def __init__(self, in_dim: int, d_model: int, patch_len: int, dropout: float = 0.1):
        super().__init__()
        self.in_dim = int(in_dim)
        self.d_model = int(d_model)
        self.patch_len = int(patch_len)
        self.value_embedding = nn.Linear(self.patch_len * self.in_dim, self.d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, L, C = x.shape
        if C != self.in_dim:
            raise ValueError(f"PatchEmbedding expected input dim {self.in_dim}, got {C}")
        if L % self.patch_len != 0:
            # Keep the most recent complete patches.
            L_new = (L // self.patch_len) * self.patch_len
            x = x[:, -L_new:, :]
            L = L_new
        P = L // self.patch_len
        x = x.reshape(B, P, self.patch_len * C)
        return self.dropout(self.value_embedding(x))


class LazyPatchEmbedding(nn.Module):
    """Patch embedding that infers the number of input features."""
    def __init__(self, d_model: int, patch_len: int, dropout: float = 0.1):
        super().__init__()
        self.d_model = int(d_model)
        self.patch_len = int(patch_len)
        self.value_embedding = nn.LazyLinear(self.d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, L, C = x.shape
        if L % self.patch_len != 0:
            L_new = (L // self.patch_len) * self.patch_len
            x = x[:, -L_new:, :]
            L = L_new
        P = L // self.patch_len
        x = x.reshape(B, P, self.patch_len * C)
        return self.dropout(self.value_embedding(x))


class LearnablePositionEmbedding(nn.Module):
    def __init__(self, max_patch_num: int, d_model: int):
        super().__init__()
        self.pos = nn.Parameter(torch.zeros(1, int(max_patch_num), int(d_model)))
        nn.init.trunc_normal_(self.pos, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pos[:, :x.size(1), :]


class ForecastHead(nn.Module):
    """Map encoder tokens [B, N_tgt, P, d_model] to [B, pred_len, N_tgt]."""

    def __init__(self, patch_num: int, d_model: int, pred_len: int,
                 dropout: float = 0.1, hidden_dim: Optional[int] = None,
                 activation: str = "gelu"):
        super().__init__()
        hidden_dim = int(hidden_dim) if hidden_dim is not None else int(d_model)
        self.flatten = nn.Flatten(start_dim=-2)
        self.feature_proj = nn.Linear(int(patch_num) * int(d_model), hidden_dim)
        self.activation = nn.GELU() if activation == "gelu" else nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(hidden_dim, int(pred_len))

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        feat = self.flatten(h)
        feat = self.feature_proj(feat)
        feat = self.activation(feat)
        feat = self.dropout(feat)
        return self.head(feat).transpose(1, 2)


# -----------------------------
# Graph network components
# -----------------------------

class SimilarityAdjacencyBuilder:
    """Build a local adjacency matrix for the stations in configs.data_path.

    adj_matrix_path accepts .npy + station_names_path, or a top-k .txt containing
    station headers and ranked similarity scores. Explicit invalid inputs raise
    errors; a complete graph is used only when no source is configured.

    Text scores define directed edges before optional max symmetrization.
    Unlisted pairs, negative weights, and diagonal entries are set to zero.
    """
    def __init__(self, root_path: Optional[str] = None, adj_matrix_path: Optional[str] = None,
                 station_names_path: Optional[str] = None, top3_similarity_path: Optional[str] = None,
                 symmetrize: bool = True):
        self.root_path = root_path
        self.adj_matrix_path = _maybe_join(root_path, adj_matrix_path)
        self.station_names_path = _maybe_join(root_path, station_names_path)
        self.top3_similarity_path = _maybe_join(root_path, top3_similarity_path)
        self.symmetrize = bool(symmetrize)

    def build(self, station_names: Sequence[str], n_nodes: int) -> np.ndarray:
        station_names = [os.path.basename(str(s).replace('\\', '/')) for s in station_names[:n_nodes]]
        if len(station_names) < n_nodes:
            if self.adj_matrix_path or self.top3_similarity_path:
                raise ValueError(f"Expected {n_nodes} graph station names in data_path, got {station_names}")
            return self._fallback(n_nodes)

        if self.adj_matrix_path:
            if not os.path.isfile(self.adj_matrix_path):
                raise FileNotFoundError(f"Graph similarity file not found: {self.adj_matrix_path}")
            extension = os.path.splitext(self.adj_matrix_path)[1].lower()
            if extension == '.txt':
                return self._from_topk_txt(station_names, n_nodes, self.adj_matrix_path)
            if extension != '.npy':
                raise ValueError(f"Graph similarity file must be .npy or top-k .txt: {self.adj_matrix_path}")
            if not self.station_names_path or not os.path.isfile(self.station_names_path):
                raise FileNotFoundError(f"A .npy graph requires station_names_path: {self.station_names_path}")
            return self._from_npy(station_names, n_nodes)

        if self.top3_similarity_path:
            if not os.path.isfile(self.top3_similarity_path):
                raise FileNotFoundError(f"Top-k similarity file not found: {self.top3_similarity_path}")
            return self._from_topk_txt(station_names, n_nodes)

        return self._fallback(n_nodes)

    def _from_npy(self, station_names: Sequence[str], n_nodes: int) -> np.ndarray:
        full_adj = np.load(self.adj_matrix_path, allow_pickle=False)
        with open(self.station_names_path, "r", encoding="utf-8-sig") as f:
            all_names = [os.path.basename(line.strip().replace('\\', '/')) for line in f if line.strip()]
        if full_adj.shape != (len(all_names), len(all_names)):
            raise ValueError(f"Graph matrix shape {full_adj.shape} does not match {len(all_names)} station names")
        name_to_idx = {name: i for i, name in enumerate(all_names)}
        try:
            idx = [name_to_idx[name] for name in station_names]
        except KeyError as e:
            warnings.warn(f"Station {e} not found in station_names_path; using fallback adjacency.")
            return self._fallback(n_nodes)
        local = np.asarray(full_adj[np.ix_(idx, idx)], dtype=np.float32)
        return self._postprocess(local)

    def _from_topk_txt(self, station_names: Sequence[str], n_nodes: int,
                       path: Optional[str] = None) -> np.ndarray:
        path = path or self.top3_similarity_path
        pair_scores: Dict[Tuple[str, str], float] = {}
        known_stations = set()
        current = None
        with open(path, "r", encoding="utf-8-sig") as f:
            for line_number, raw in enumerate(f, 1):
                line = raw.strip()
                if not line or line.startswith('#'):
                    continue
                if line.endswith(":") and not re.match(r"^\d+\.", line):
                    current = os.path.basename(line[:-1].strip().replace('\\', '/'))
                    if not current:
                        raise ValueError(f"Empty station header in {path}:{line_number}")
                    known_stations.add(current)
                    continue
                m = re.fullmatch(r"\d+\.\s+(.+):\s*(\S+)", line)
                if not m or current is None:
                    raise ValueError(f"Invalid top-k similarity record in {path}:{line_number}: {line!r}")
                other = os.path.basename(m.group(1).strip().replace('\\', '/'))
                try:
                    score = float(m.group(2))
                except ValueError as exc:
                    raise ValueError(f"Invalid similarity score in {path}:{line_number}: {m.group(2)!r}") from exc
                if not other or not np.isfinite(score):
                    raise ValueError(f"Invalid station or non-finite similarity in {path}:{line_number}")
                known_stations.add(other)
                key = (current, other)
                pair_scores[key] = max(pair_scores.get(key, score), score)

        if not pair_scores:
            raise ValueError(f"No ranked similarity records found in {path}")
        missing = sorted(set(station_names) - known_stations)
        if missing:
            raise ValueError(f"Graph stations missing from top-k similarity file {path}: {missing}")

        local = np.zeros((n_nodes, n_nodes), dtype=np.float32)
        for i, a in enumerate(station_names):
            for j, b in enumerate(station_names):
                local[i, j] = pair_scores.get((a, b), 0.0)
        return self._postprocess(local)

    def _fallback(self, n_nodes: int) -> np.ndarray:
        # Connect every pair of distinct stations when no adjacency source is available.
        local = np.ones((n_nodes, n_nodes), dtype=np.float32)
        return self._postprocess(local)

    def _postprocess(self, adj: np.ndarray) -> np.ndarray:
        adj = np.asarray(adj, dtype=np.float32)
        if self.symmetrize:
            adj = np.maximum(adj, adj.T)
        adj = np.nan_to_num(adj, nan=0.0, posinf=0.0, neginf=0.0)
        adj = np.maximum(adj, 0.0)
        np.fill_diagonal(adj, 0.0)
        return adj


class GraphTokenPropagation(nn.Module):
    """Similarity-guided graph propagation over station patch tokens.

    Input:
        z:   [B, N, P, D]
        adj: [N, N] or [B, N, N]
    Output:
        out: [B, N, P, D]

    This layer keeps the graph topology/weights fixed during a forward pass, but uses
    learnable self/neighbor transformations on patch-level station representations.
    """
    def __init__(self, d_model: int, dropout: float = 0.1, add_residual: bool = True):
        super().__init__()
        self.d_model = int(d_model)
        self.add_residual = bool(add_residual)
        self.self_proj = nn.Linear(d_model, d_model)
        self.neigh_proj = nn.Linear(d_model, d_model)
        self.gate = nn.Sequential(
            nn.Linear(2 * d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
            nn.Sigmoid(),
        )
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_model)

    @staticmethod
    def _normalize_adj(adj: torch.Tensor) -> torch.Tensor:
        # Exclude diagonal entries before normalizing supplied or restored adjacency.
        diagonal = torch.eye(adj.size(-1), dtype=torch.bool, device=adj.device)
        adj = adj.masked_fill(diagonal, 0.0).clamp_min(0.0)
        denom = adj.sum(dim=-1, keepdim=True).clamp_min(1e-6)
        return adj / denom

    def forward(self, z: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        B, N, P, D = z.shape
        if adj.dim() == 2:
            adj = adj.unsqueeze(0).expand(B, -1, -1)
        if adj.shape[-2:] != (N, N):
            raise ValueError(f"adj shape must end with ({N}, {N}), got {tuple(adj.shape)}")
        adj = self._normalize_adj(adj).to(device=z.device, dtype=z.dtype)

        # Aggregate neighbor station tokens at each patch step.
        # adj[b, i, j] = weight from station j into station i.
        neigh = torch.einsum("bij,bjpd->bipd", adj, z)  # [B, N, P, D]

        gate = self.gate(torch.cat([z, neigh], dim=-1))
        msg = self.self_proj(z) + gate * self.neigh_proj(neigh)
        msg = self.dropout(msg)
        out = z + msg if self.add_residual else msg
        return self.norm(out)


# -----------------------------
# encoder
# -----------------------------

class LiteEncoderLayer(nn.Module):
    """Power self-attention, Meteorology cross-attention, and a feed-forward network."""
    def __init__(self, d_model: int, n_heads: int, d_ff: Optional[int] = None,
                 dropout: float = 0.1, activation: str = "gelu"):
        super().__init__()
        d_ff = d_ff or 4 * d_model
        self.self_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.cross_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU() if activation == "gelu" else nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, power: torch.Tensor, meteorology: torch.Tensor) -> torch.Tensor:
        power_sa, _ = self.self_attn(power, power, power, need_weights=False)
        power = self.norm1(power + self.dropout(power_sa))

        power_ca, _ = self.cross_attn(power, meteorology, meteorology, need_weights=False)
        power = self.norm2(power + self.dropout(power_ca))

        return self.norm3(power + self.ffn(power))


class LiteEncoder(nn.Module):
    def __init__(self, layers: Sequence[nn.Module], norm_layer: Optional[nn.Module] = None):
        super().__init__()
        self.layers = nn.ModuleList(layers)
        self.norm = norm_layer

    def forward(self, power: torch.Tensor, meteorology: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            power = layer(power, meteorology)
        if self.norm is not None:
            power = self.norm(power)
        return power


# -----------------------------
# Main model
# -----------------------------

class Model(nn.Module):
    """Combine power and Meteorology tokens for short- or long-term forecasting.

    MultiPower enables the graph by default. In graph mode, the power branch
    sends the main station's graph-enhanced tokens to the attention encoder.
    With 4D input, the Meteorology branch takes context from one reference station.
    """

    def __init__(self, configs):
        super().__init__()
        self.configs = configs
        self.task_name = configs.task_name
        self.features = configs.features
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.use_norm = getattr(configs, "use_norm", True)
        self.patch_len = configs.patch_len
        self.patch_num = self.seq_len // self.patch_len
        self.enc_in = configs.enc_in
        self.d_model = configs.d_model
        self.data = getattr(configs, "data", None)
        self.root_path = getattr(configs, "root_path", None)
        self.data_path = getattr(configs, "data_path", None)

        # Activate graph by default only for MultiPower.
        self.use_graph = _as_bool(getattr(configs, "use_graph", self.data == "MultiPower"))
        self.graph_num_nodes = int(getattr(configs, "graph_num_nodes", 4))
        self.main_node_index = int(getattr(configs, "main_node_index", 0))

        # In MultiPower, default x_enc channels are assumed:
        #   [main_power, aux1_power, aux2_power, aux3_power, shared_weather...]
        # For 4D MultiPower input, these are station indices along x_enc[:, :, station, feature].
        # For the older 3D flattened input, these are channel indices. Negative values are
        # resolved at forward time because the relevant dimension differs for 3D and 4D.
        self.graph_power_indices = _parse_indices(
            getattr(configs, "graph_power_indices", None),
            default=list(range(self.graph_num_nodes)),
            modulo=None,
        )
        if len(self.graph_power_indices) < self.graph_num_nodes:
            raise ValueError(
                f"graph_power_indices must contain at least graph_num_nodes={self.graph_num_nodes} indices; "
                f"got {self.graph_power_indices}"
            )
        self.graph_power_indices = self.graph_power_indices[:self.graph_num_nodes]
        self.main_power_index = self.graph_power_indices[self.main_node_index]

        # In 4D Dataset_Power_MultiInput style, choose which feature inside each station
        # is PV power. Defaults to -1 to match the usual MS convention where target is last.
        self.power_feature_index = int(getattr(configs, "power_feature_index", -1))
        # Read the existing config keys for the Meteorology branch.
        self.meteorology_station_index = int(getattr(configs, "exog_station_index", self.main_node_index))
        self.meteorology_feature_indices = getattr(configs, "exog_feature_indices", None)

        # Compute solar elevation from time marks unless a column is configured.
        self.use_solar_elevation = _as_bool(getattr(configs, "SolarElevation", False))
        self.solar_latitude_deg = -23.7
        self.solar_time_feature_indices = _parse_indices(
            getattr(configs, "solar_time_feature_indices", None),
            default=[0, 1, 2, 3, 4],
            modulo=None,
        )
        self.solar_elevation_index = getattr(configs, "solar_elevation_index", None)
        self.solar_elevation_mark_index = getattr(configs, "solar_elevation_mark_index", None)

        # Branch inputs.
        self.target_indices = self._resolve_target_indices(configs)
        self.meteorology_indices = self._resolve_meteorology_indices(configs)
        self.n_targets = len(self.target_indices)

        # Patch embeddings.
        self.power_patch_embedding = PatchEmbedding(
            in_dim=1,
            d_model=configs.d_model,
            patch_len=configs.patch_len,
            dropout=configs.dropout,
        )
        self.power_pos_embedding = LearnablePositionEmbedding(self.patch_num, configs.d_model)

        self.meteorology_patch_embedding = LazyPatchEmbedding(
            d_model=configs.d_model,
            patch_len=configs.patch_len,
            dropout=configs.dropout,
        )
        self.meteorology_pos_embedding = LearnablePositionEmbedding(self.patch_num, configs.d_model)

        if self.use_solar_elevation:
            self.solar_patch_embedding = LazyPatchEmbedding(
                d_model=configs.d_model,
                patch_len=configs.patch_len,
                dropout=configs.dropout,
            )
            self.solar_pos_embedding = LearnablePositionEmbedding(self.patch_num, configs.d_model)
            self.solar_gate = nn.Sequential(
                nn.Linear(configs.d_model, configs.d_model),
                nn.Sigmoid(),
            )
        else:
            self.solar_patch_embedding = None
            self.solar_pos_embedding = None
            self.solar_gate = None

        # Graph network.
        if self.use_graph:
            self.graph_layer = GraphTokenPropagation(
                d_model=configs.d_model,
                dropout=configs.dropout,
                add_residual=_as_bool(getattr(configs, "graph_residual", True)),
            )
            adj = self._build_initial_adjacency(configs)
            self.register_buffer("graph_adj", torch.tensor(adj, dtype=torch.float32), persistent=True)
        else:
            self.graph_layer = None
            self.register_buffer("graph_adj", torch.zeros(1, 1, dtype=torch.float32), persistent=False)

        # encoder.
        self.encoder = LiteEncoder(
            [
                LiteEncoderLayer(
                    d_model=configs.d_model,
                    n_heads=configs.n_heads,
                    d_ff=configs.d_ff,
                    dropout=configs.dropout,
                    activation=getattr(configs, "activation", "gelu"),
                )
                for _ in range(configs.e_layers)
            ],
            norm_layer=nn.LayerNorm(configs.d_model),
        )

        self.head = ForecastHead(
            patch_num=self.patch_num,
            d_model=configs.d_model,
            pred_len=configs.pred_len,
            dropout=configs.dropout,
            hidden_dim=getattr(configs, "head_hidden_dim", configs.d_model),
            activation=getattr(configs, "activation", "gelu"),
        )

    # ---------- config/index helpers ----------

    def _resolve_target_indices(self, configs) -> List[int]:
        explicit = getattr(configs, "target_indices", None)
        if explicit is not None:
            return _parse_indices(explicit, modulo=self.enc_in)
        if self.use_graph:
            # In graph mode the model outputs the main station only.
            # For 4D input, this value is only used by 3D fallback denormalization.
            return [self.main_power_index % max(1, self.enc_in)]
        if self.features == "M":
            return list(range(self.enc_in))
        return [self.enc_in - 1]

    def _resolve_meteorology_indices(self, configs) -> List[int]:
        explicit = getattr(configs, "exog_indices", None)
        if explicit is not None:
            return _parse_indices(explicit, modulo=self.enc_in)
        # With 4D input, these indices select features within the reference station.
        if self.use_graph:
            if self.meteorology_feature_indices is not None:
                return _parse_indices(self.meteorology_feature_indices, modulo=self.enc_in)
            power_idx = self.power_feature_index % max(1, self.enc_in)
            return [i for i in range(self.enc_in) if i != power_idx]
        if self.features == "M":
            return list(range(self.enc_in))
        target_set = set(self.target_indices)
        return [i for i in range(self.enc_in) if i not in target_set]

    def _build_initial_adjacency(self, configs) -> np.ndarray:
        # data_path is expected to be ordered as target;aux1;aux2;aux3.
        # Extract the .npy/.txt subgraph in exactly the current input station order.
        all_data_path_names = _split_data_path(getattr(configs, "data_path", None))
        if len(all_data_path_names) >= max(self.graph_power_indices[:self.graph_num_nodes]) + 1:
            station_names = [all_data_path_names[i] for i in self.graph_power_indices[:self.graph_num_nodes]]
        else:
            station_names = all_data_path_names[:self.graph_num_nodes]

        builder = SimilarityAdjacencyBuilder(
            root_path=getattr(configs, "root_path", None),
            adj_matrix_path=getattr(configs, "adj_matrix_path", getattr(configs, "adjacency_path", None)),
            station_names_path=getattr(configs, "station_names_path", None),
            top3_similarity_path=getattr(configs, "top3_similarity_path", None),
            symmetrize=_as_bool(getattr(configs, "graph_symmetrize", True)),
        )
        return builder.build(station_names, self.graph_num_nodes)

    # ---------- normalization helpers ----------

    def _normalize(self, x_enc: torch.Tensor):
        if not self.use_norm:
            return x_enc, None, None
        means = x_enc.mean(1, keepdim=True).detach()
        x = x_enc - means
        stdev = torch.sqrt(torch.var(x, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x = x / stdev
        return x, means, stdev

    def _denormalize_targets(self, dec_out: torch.Tensor, means: Optional[torch.Tensor], stdev: Optional[torch.Tensor]):
        if not self.use_norm:
            return dec_out
        if means.dim() == 4:
            # 4D MultiPower: means/stdev are [B, 1, n_files, n_features].
            main_station = self.graph_power_indices[self.main_node_index] % means.shape[2]
            power_idx = self.power_feature_index % means.shape[3]
            target_means = means[:, :, main_station, power_idx].unsqueeze(-1)  # [B, 1, 1]
            target_stdev = stdev[:, :, main_station, power_idx].unsqueeze(-1)  # [B, 1, 1]
            return dec_out * target_stdev.repeat(1, self.pred_len, 1) + target_means.repeat(1, self.pred_len, 1)

        tgt = torch.tensor(self.target_indices, dtype=torch.long, device=dec_out.device)
        target_means = means[:, :, tgt]   # [B, 1, N_tgt]
        target_stdev = stdev[:, :, tgt]   # [B, 1, N_tgt]
        return dec_out * target_stdev.repeat(1, self.pred_len, 1) + target_means.repeat(1, self.pred_len, 1)

    # ---------- input builders ----------

    def _make_meteorology_input(self, x_enc: torch.Tensor, x_mark_enc: Optional[torch.Tensor]) -> torch.Tensor:
        parts = []
        if x_enc.dim() == 4:
            station_idx = self.meteorology_station_index % x_enc.shape[2]
            x_ref = x_enc[:, :, station_idx, :]
        else:
            x_ref = x_enc

        if self.meteorology_indices:
            parts.append(_select_last_dim(x_ref, self.meteorology_indices))

        if x_mark_enc is not None:
            if x_mark_enc.dim() == 4:
                mark_station = self.meteorology_station_index % x_mark_enc.shape[2]
                parts.append(x_mark_enc[:, :, mark_station, :])
            else:
                parts.append(x_mark_enc)

        if len(parts) == 0:
            B, L = x_enc.shape[0], x_enc.shape[1]
            parts.append(torch.zeros(B, L, 1, device=x_enc.device, dtype=x_enc.dtype))
        return torch.cat(parts, dim=-1)

    def _make_solar_input(self, x_enc: torch.Tensor, x_mark_enc: Optional[torch.Tensor]) -> torch.Tensor:
        """Return solar elevation sequence [B, L, 1].

        Read a configured precomputed column or calculate elevation from time marks.
        """
        if self.solar_elevation_index is not None:
            if x_enc.dim() == 4:
                station_idx = self.meteorology_station_index % x_enc.shape[2]
                idx = int(self.solar_elevation_index) % x_enc.shape[-1]
                return x_enc[:, :, station_idx, idx:idx + 1]
            idx = int(self.solar_elevation_index) % x_enc.shape[-1]
            return x_enc[:, :, idx:idx + 1]

        if x_mark_enc is None:
            raise ValueError(
                "configs.SolarElevation=True requires x_mark_enc unless "
                "solar_elevation_index selects a precomputed column in x_enc."
            )

        if self.solar_elevation_mark_index is not None:
            idx = int(self.solar_elevation_mark_index) % x_mark_enc.shape[-1]
            if x_mark_enc.dim() == 4:
                station_idx = self.meteorology_station_index % x_mark_enc.shape[2]
                return x_mark_enc[:, :, station_idx, idx:idx + 1]
            return x_mark_enc[:, :, idx:idx + 1]

        if x_mark_enc.dim() == 4:
            station_idx = self.meteorology_station_index % x_mark_enc.shape[2]
            time_mark = x_mark_enc[:, :, station_idx, :]
        else:
            time_mark = x_mark_enc

        if time_mark.shape[-1] < 5:
            raise ValueError(
                f"SolarElevation requires at least 5 time features in x_mark_enc, got {time_mark.shape[-1]}. "
                "Expected [MinuteOfHour, HourOfDay, DayOfWeek, DayOfMonth, DayOfYear] in [-0.5, 0.5]."
            )

        time_indices = [i % time_mark.shape[-1] for i in self.solar_time_feature_indices]
        if len(time_indices) < 5:
            raise ValueError(
                f"solar_time_feature_indices must contain 5 columns, got {time_indices}."
            )
        time_features = time_mark[:, :, time_indices[:5]]

        solar = _estimate_solar_elevation_from_time_features(time_features, latitude_deg=self.solar_latitude_deg)

        return solar.unsqueeze(-1).to(device=x_enc.device, dtype=x_enc.dtype)

    # ---------- branch encoders ----------

    def _encode_power(self, x_enc: torch.Tensor, graph_adj=None) -> torch.Tensor:
        """Return patch tokens for the main graph station or the selected targets."""
        if self.use_graph:
            if x_enc.dim() == 4:
                # x_enc: [B, L, n_files, n_features]. Read only PV power feature
                # from station dimension for the graph branch.
                station_dim = x_enc.shape[2]
                station_indices = [i % station_dim for i in self.graph_power_indices[:self.graph_num_nodes]]
                power_idx = self.power_feature_index % x_enc.shape[3]
                y_all_nodes = x_enc[:, :, station_indices, power_idx]  # [B, L, N_graph]
            else:
                # Backward-compatible flattened 3D mode.
                y_all_nodes = _select_last_dim(x_enc, self.graph_power_indices[:self.graph_num_nodes])  # [B, L, N_graph]

            station_tokens = []
            for k in range(self.graph_num_nodes):
                z = self.power_patch_embedding(y_all_nodes[:, :, k:k + 1])  # [B, P, D]
                z = self.power_pos_embedding(z)
                station_tokens.append(z)
            z_nodes = torch.stack(station_tokens, dim=1)  # [B, N_graph, P, D]
            z_nodes = self.graph_layer(z_nodes, self.graph_adj if graph_adj is None else graph_adj)
            return z_nodes[:, self.main_node_index:self.main_node_index + 1, :, :]  # [B, 1, P, D]

        # Non-graph behavior: target variable(s) only.
        if x_enc.dim() == 4:
            station_idx = self.meteorology_station_index % x_enc.shape[2]
            x_ref = x_enc[:, :, station_idx, :]
            y = _select_last_dim(x_ref, self.target_indices)
        else:
            y = _select_last_dim(x_enc, self.target_indices)  # [B, L, N_tgt]
        target_tokens = []
        for k in range(self.n_targets):
            z = self.power_patch_embedding(y[:, :, k:k + 1])
            z = self.power_pos_embedding(z)
            target_tokens.append(z)
        return torch.stack(target_tokens, dim=1)  # [B, N_tgt, P, D]

    def _apply_solar_elevation(self, z_y: torch.Tensor, x_enc: torch.Tensor, x_mark_enc: Optional[torch.Tensor]) -> torch.Tensor:
        if not self.use_solar_elevation:
            return z_y
        solar = self._make_solar_input(x_enc, x_mark_enc)  # [B, L, 1]
        z_s = self.solar_patch_embedding(solar)            # [B, P, D]
        z_s = self.solar_pos_embedding(z_s)
        gate = self.solar_gate(z_s).unsqueeze(1)           # [B, 1, P, D]
        return z_y + gate * z_s.unsqueeze(1)

    # ---------- forecast ----------

    def forecast_lite(self, x_enc: torch.Tensor, x_mark_enc: Optional[torch.Tensor],
                      x_dec: Optional[torch.Tensor] = None, x_mark_dec: Optional[torch.Tensor] = None,
                      graph_adj=None) -> torch.Tensor:
        x_enc, means, stdev = self._normalize(x_enc)

        power_tokens = self._encode_power(x_enc, graph_adj=graph_adj)
        power_tokens = self._apply_solar_elevation(power_tokens, x_enc, x_mark_enc)

        meteorology_input = self._make_meteorology_input(x_enc, x_mark_enc)
        meteorology_tokens = self.meteorology_patch_embedding(meteorology_input)
        meteorology_tokens = self.meteorology_pos_embedding(meteorology_tokens)

        # Cross-attention is applied per target variable.
        B, n_targets, P, D = power_tokens.shape
        power_flat = power_tokens.reshape(B * n_targets, P, D)
        meteorology_per_target = meteorology_tokens.repeat_interleave(n_targets, dim=0)
        enc_out = self.encoder(power_flat, meteorology_per_target)
        enc_out = enc_out.reshape(B, n_targets, P, D)

        # Head predicts normalized target, then denormalize by target/main station stats.
        dec_out = self.head(enc_out)  # [B, pred_len, N_tgt]
        dec_out = self._denormalize_targets(dec_out, means, stdev)
        return dec_out

    def forward(self, x_enc: torch.Tensor, x_mark_enc: Optional[torch.Tensor],
                x_dec: Optional[torch.Tensor], x_mark_dec: Optional[torch.Tensor], mask=None,
                graph_adj=None) -> Optional[torch.Tensor]:
        if self.task_name in ["long_term_forecast", "short_term_forecast"]:
            dec_out = self.forecast_lite(x_enc, x_mark_enc, x_dec, x_mark_dec, graph_adj=graph_adj)
            return dec_out[:, -self.pred_len:, :]
        return None
