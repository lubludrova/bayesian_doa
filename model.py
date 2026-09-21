from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import itertools
import math
import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
METHODS = ("physical_rqmc", "quotient_self_normalized")
@dataclass(frozen=True)
class DoASpec:
    sensors: int = 8
    fov_deg: float = 60.0
    spacing_values: tuple[float, ...] = (0.5, 0.75, 1.0)
    snapshot_values: tuple[int, ...] = (4, 16, 64)
    snr_db_values: tuple[float, ...] = (-15.0, -10.0, -5.0, 0.0, 10.0)

    @property
    def u_limit(self) -> float:
        return math.sin(math.radians(self.fov_deg))

    @property
    def context_dim(self) -> int:
        return 2 * self.sensors * self.sensors + 4


def sample_covariance(observations: torch.Tensor) -> torch.Tensor:
    snapshots = observations.shape[1]
    return torch.einsum("btm,btn->bmn", observations, observations.conj()) / snapshots
@dataclass
class MultiSourceBatch:
    observations: torch.Tensor
    target_w: torch.Tensor
    spacing: torch.Tensor
    snr_db: torch.Tensor
    snapshots: int
    sensor_offsets: torch.Tensor | None = None


def log_likelihood_on_pair_grid(
    batch: MultiSourceBatch,
    pairs_w: torch.Tensor,
    spec: DoASpec,
) -> torch.Tensor:


    covariance = sample_covariance(batch.observations)
    real_dtype = covariance.real.dtype
    pairs_w = pairs_w.to(device=covariance.device, dtype=real_dtype)
    sensor_index = torch.arange(
        spec.sensors, device=covariance.device, dtype=real_dtype
    )
    direction_cosine = pairs_w * spec.u_limit
    positions = batch.spacing[:, None] * sensor_index
    if batch.sensor_offsets is not None:
        offsets = batch.sensor_offsets.to(device=covariance.device, dtype=real_dtype)
        if offsets.ndim == 1:
            offsets = offsets[None]
        positions = positions + offsets
    phase = (
        -2.0
        * math.pi
        * direction_cosine[None, :, :, None]
        * positions[:, None, None, :]
    )
    steering = torch.exp(1j * phase).to(covariance.dtype).permute(0, 1, 3, 2)

    gram = torch.einsum("bpmi,bpmj->bpij", steering.conj(), steering)
    snr = (0.5 * torch.pow(10.0, batch.snr_db / 10.0)).to(real_dtype)
    eye = torch.eye(2, device=covariance.device, dtype=covariance.dtype)
    small = eye[None, None] + snr[:, None, None, None] * gram
    _, logdet = torch.linalg.slogdet(small)
    inverse = torch.linalg.inv(small)
    projected = torch.einsum(
        "bpmi,bmn,bpnj->bpij", steering.conj(), covariance, steering
    )
    correction = torch.einsum("bpij,bpji->bp", inverse, projected).real
    trace = covariance.diagonal(dim1=-2, dim2=-1).real.sum(dim=-1, keepdim=True)
    quadratic = trace - snr[:, None] * correction
    return -float(batch.snapshots) * (logdet.real + quadratic)
@dataclass
class ThreeSourceBatch:
    observations: torch.Tensor
    target_w: torch.Tensor
    spacing: torch.Tensor
    snr_db: torch.Tensor
    snapshots: int


def log_likelihood_on_triple_grid(
    batch: ThreeSourceBatch,
    triples_w: torch.Tensor,
    spec: DoASpec,
) -> torch.Tensor:
    covariance = sample_covariance(batch.observations)
    real_dtype = covariance.real.dtype
    triples_w = triples_w.to(device=covariance.device, dtype=real_dtype)
    sensor_index = torch.arange(
        spec.sensors, device=covariance.device, dtype=real_dtype
    )
    direction_cosine = triples_w * spec.u_limit
    positions = batch.spacing[:, None] * sensor_index
    phase = (
        -2.0
        * math.pi
        * direction_cosine[None, :, :, None]
        * positions[:, None, None, :]
    )
    steering = torch.exp(1j * phase).to(covariance.dtype).permute(0, 1, 3, 2)
    gram = torch.einsum("bpmi,bpmj->bpij", steering.conj(), steering)
    rho = (torch.pow(10.0, batch.snr_db / 10.0) / 3.0).to(real_dtype)
    eye = torch.eye(3, device=covariance.device, dtype=covariance.dtype)
    small = eye[None, None] + rho[:, None, None, None] * gram
    _, logdet = torch.linalg.slogdet(small)
    inverse = torch.linalg.inv(small)
    projected = torch.einsum(
        "bpmi,bmn,bpnj->bpij", steering.conj(), covariance, steering
    )
    correction = torch.einsum("bpij,bpji->bp", inverse, projected).real
    trace = covariance.diagonal(dim1=-2, dim2=-1).real.sum(dim=-1, keepdim=True)
    quadratic = trace - rho[:, None] * correction
    return -float(batch.snapshots) * (logdet.real + quadratic)

@dataclass
class FourSourceBatch:
    observations: torch.Tensor
    target_w: torch.Tensor
    spacing: torch.Tensor
    snr_db: torch.Tensor
    snapshots: int


def log_likelihood_on_quadruple_grid(
    batch: FourSourceBatch,
    quadruples_w: torch.Tensor,
    spec: DoASpec,
) -> torch.Tensor:
    covariance = sample_covariance(batch.observations)
    real_dtype = covariance.real.dtype
    quadruples_w = quadruples_w.to(device=covariance.device, dtype=real_dtype)
    sensor_index = torch.arange(
        spec.sensors, device=covariance.device, dtype=real_dtype
    )
    direction_cosine = quadruples_w * spec.u_limit
    positions = batch.spacing[:, None] * sensor_index
    phase = (
        -2.0
        * math.pi
        * direction_cosine[None, :, :, None]
        * positions[:, None, None, :]
    )
    steering = torch.exp(1j * phase).to(covariance.dtype).permute(0, 1, 3, 2)
    gram = torch.einsum("bpmi,bpmj->bpij", steering.conj(), steering)
    rho = (torch.pow(10.0, batch.snr_db / 10.0) / 4.0).to(real_dtype)
    eye = torch.eye(4, device=covariance.device, dtype=covariance.dtype)
    small = eye[None, None] + rho[:, None, None, None] * gram
    _, logdet = torch.linalg.slogdet(small)
    inverse = torch.linalg.inv(small)
    projected = torch.einsum(
        "bpmi,bmn,bpnj->bpij", steering.conj(), covariance, steering
    )
    correction = torch.einsum("bpij,bpji->bp", inverse, projected).real
    trace = covariance.diagonal(dim1=-2, dim2=-1).real.sum(dim=-1, keepdim=True)
    quadratic = trace - rho[:, None] * correction
    return -float(batch.snapshots) * (logdet.real + quadratic)


Batch = MultiSourceBatch | ThreeSourceBatch | FourSourceBatch

def require_ideal_ula(sensor_offsets: torch.Tensor | None) -> None:


    if sensor_offsets is None:
        return
    if not bool(torch.isfinite(sensor_offsets).all()) or bool(
        (sensor_offsets != 0).any()
    ):
        raise ValueError("exact quotient reduction requires an ideal ULA")


def canonical_spacing(spec: DoASpec) -> float:


    return 1.0 / (2.0 * spec.u_limit)


def quotient_coordinate(
    w: torch.Tensor, spacing: torch.Tensor, spec: DoASpec
) -> torch.Tensor:


    if spacing.numel() == 0 or not bool(torch.isfinite(spacing).all()):
        raise ValueError("spacing must be finite and non-empty")
    if bool((spacing <= 0).any()):
        raise ValueError("spacing must be positive")

    spatial_frequency = spacing * spec.u_limit * w
    wrapped = torch.remainder(spatial_frequency + 0.5, 1.0) - 0.5
    return 2.0 * wrapped
def canonical_batch(batch: Batch, spec: DoASpec) -> Batch:


    if isinstance(batch, MultiSourceBatch):
        require_ideal_ula(batch.sensor_offsets)
        return MultiSourceBatch(
            batch.observations,
            batch.target_w,
            torch.full_like(batch.spacing, canonical_spacing(spec)),
            batch.snr_db,
            batch.snapshots,
        )
    return type(batch)(
        batch.observations,
        batch.target_w,
        torch.full_like(batch.spacing, canonical_spacing(spec)),
        batch.snr_db,
        batch.snapshots,
    )


@torch.no_grad()
def likelihood_in_chunks(
    batch: Batch,
    candidates: torch.Tensor,
    spec: DoASpec,
    *,
    chunk_size: int,
) -> torch.Tensor:


    if chunk_size < 1:
        raise ValueError("chunk_size must be positive")
    pieces = []
    for start in range(0, len(candidates), chunk_size):
        chunk = candidates[start : start + chunk_size]
        if isinstance(batch, MultiSourceBatch):
            likelihood = log_likelihood_on_pair_grid(batch, chunk, spec)
        elif isinstance(batch, ThreeSourceBatch):
            likelihood = log_likelihood_on_triple_grid(batch, chunk, spec)
        else:
            likelihood = log_likelihood_on_quadruple_grid(batch, chunk, spec)
        pieces.append(likelihood[0])
    return torch.cat(pieces)
def prior_density(points: torch.Tensor, beta: float) -> torch.Tensor:
    if beta and (points.shape[1] != 2 or abs(beta) >= 1):
        raise ValueError("the correlated prior requires K=2 and |beta|<1")
    return 1.0 + beta * points.prod(dim=-1)


def branch_rows(points: torch.Tensor, scale: float):

    order = math.floor((scale + 1) / 2)
    for shifts in itertools.product(range(-order, order + 1), repeat=points.shape[1]):
        physical = (points + 2 * torch.tensor(shifts, dtype=points.dtype)) / scale

        valid = ((physical >= -1) & (physical < 1)).all(dim=1)
        if bool(valid.any()):
            yield valid, physical[valid].sort(dim=1).values


def quotient_ratio(points: torch.Tensor, scale: float, beta: float) -> torch.Tensor:

    order = math.floor((scale + 1) / 2)
    shifts = torch.arange(-order, order + 1, dtype=points.dtype)
    branches = (points[:, :, None] + 2 * shifts) / scale
    valid = (branches >= -1) & (branches < 1)
    counts = valid.sum(dim=2).to(points.dtype)
    mass = counts.prod(dim=1)
    if beta:
        if points.shape[1] != 2:
            raise ValueError("correlated quotient prior requires K=2")
        mass += beta * torch.where(valid, branches, 0).sum(dim=2).prod(dim=1)
    return mass / scale ** points.shape[1]
def sobol_nodes(sources, seed, points, device="cpu"):

    if sources not in (2, 3, 4) or not isinstance(points, int) or points < 1:
        raise ValueError("K=2/3/4 and a positive integer point count are required")
    if not isinstance(seed, int) or seed < 0:
        raise ValueError("the Sobol seed must be a nonnegative integer")
    uniform = torch.quasirandom.SobolEngine(sources, scramble=True, seed=seed).draw(
        points, dtype=torch.float64
    )
    return (2 * uniform - 1).sort(dim=1).values.to(device)


def _check_points(points):
    if (points.ndim != 2 or len(points) == 0 or points.shape[1] not in (2, 3, 4)
            or points.dtype != torch.float64):
        raise ValueError("points must be a nonempty float64 [N,K] tensor, K=2/3/4")
    if (not bool(torch.isfinite(points).all()) or bool((points.abs() > 1).any())
            or bool((points.diff(dim=1) < 0).any())):
        raise ValueError("points must be finite, sorted and lie in [-1,1]")


@torch.no_grad()
def evaluate_nodes(batch, spec, nodes, method, chunk_size=65536):

    if method not in METHODS:
        raise ValueError("unsupported integration method: " + str(method))
    _check_points(nodes)
    if not isinstance(chunk_size, int) or chunk_size < 1:
        raise ValueError("chunk_size must be a positive integer")
    if not isinstance(batch, (MultiSourceBatch, ThreeSourceBatch, FourSourceBatch)):
        raise TypeError("expected a two-, three-, or four-source observation batch")
    sources = {MultiSourceBatch: 2, ThreeSourceBatch: 3, FourSourceBatch: 4}[type(batch)]
    count = len(batch.observations)
    if (batch.observations.ndim != 3 or count == 0
            or batch.target_w.shape != (count, sources)
            or batch.spacing.shape != (count,) or batch.snr_db.shape != (count,)
            or batch.observations.shape[1:] != (batch.snapshots, spec.sensors)
            or nodes.shape[1] != sources):
        raise ValueError("observation batch and node shapes do not agree")
    if (batch.observations.dtype != torch.complex128
            or any(x.dtype != torch.float64 for x in (batch.target_w, batch.spacing, batch.snr_db))
            or any(x.device != nodes.device for x in
                   (batch.observations, batch.target_w, batch.spacing, batch.snr_db))):
        raise ValueError("move_batch and nodes must use the same device and double precision")
    if method == "quotient_self_normalized" and bool((nodes >= 1).any()):
        raise ValueError("canonical nodes must lie in the half-open interval [-1,1)")
    likelihood_batch = batch if method == "physical_rqmc" else canonical_batch(batch, spec)
    likelihood = {
        2: log_likelihood_on_pair_grid,
        3: log_likelihood_on_triple_grid,
        4: log_likelihood_on_quadruple_grid,
    }[sources]
    values = [likelihood(likelihood_batch, nodes[start:start + chunk_size], spec)
              for start in range(0, len(nodes), chunk_size)]
    result = torch.cat(values, dim=1)
    if not bool(torch.isfinite(result).all()):
        raise RuntimeError("nonfinite node log likelihood")
    return result

def load_observation(index):
    with np.load(ROOT / "data/observations.npz") as saved:
        sources = int(saved["sources"][index])
        snapshots = int(saved["snapshots"][index])
        spec = DoASpec(sensors=8, fov_deg=float(saved["fov_deg"][index]))
        batch_type = {
            2: MultiSourceBatch,
            3: ThreeSourceBatch,
            4: FourSourceBatch,
        }[sources]
        batch = batch_type(
            torch.from_numpy(saved["observations"][index:index+1, :snapshots].copy()),
            torch.from_numpy(saved["target_w"][index:index+1, :sources].copy()),
            torch.from_numpy(saved["spacing"][index:index+1].copy()),
            torch.from_numpy(saved["snr_db"][index:index+1].copy()),
            snapshots,
        )
        cell = int(saved["cell"][index])
        episode = int(saved["episode"][index])
    return spec, batch, sources, cell, episode
