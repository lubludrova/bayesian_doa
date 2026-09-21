import argparse
import itertools
import json
import math
from pathlib import Path
import numpy as np
import torch
from model import (load_observation, branch_rows, prior_density, quotient_ratio,
                   quotient_coordinate, canonical_batch, likelihood_in_chunks)

SEED = 20260909
RANDOM_POINTS = 256
DELTA = 1e-9
TOLERANCES = {
    "max_log_likelihood_error": 1e-10,
    "max_log_density_error": 1e-10,
    "max_relative_H_error": 1e-12,
    "max_kappa_error": 1e-12,
    "max_kappa_sum_error": 1e-12,
    "max_coordinate_error": 1e-12,
}
def direct_covariance_log_likelihood(
    observations: np.ndarray,
    candidates: np.ndarray,
    *,
    spacing: float,
    snr_db: float,
    u_limit: float,
) -> np.ndarray:


    observations = np.asarray(observations, dtype=np.complex128)
    candidates = np.asarray(candidates, dtype=np.float64)
    if observations.ndim != 2 or candidates.ndim != 2:
        raise ValueError("observations and candidates must both be matrices")
    snapshots, sensors = observations.shape
    sources = candidates.shape[1]
    if snapshots < 1 or sensors < 1 or sources < 1:
        raise ValueError("observations and candidates must be non-empty")
    if not np.isfinite(candidates).all() or np.any(np.abs(candidates) > 1.0):
        raise ValueError("candidates must be finite and lie in [-1, 1]")

    sample_covariance = observations.T @ observations.conj() / snapshots
    sensor_index = np.arange(sensors, dtype=np.float64)
    rho = 10.0 ** (float(snr_db) / 10.0) / sources
    identity = np.eye(sensors, dtype=np.complex128)
    values = np.empty(len(candidates), dtype=np.float64)

    for index, directions in enumerate(candidates):
        phase = (
            -2.0
            * np.pi
            * float(spacing)
            * float(u_limit)
            * sensor_index[:, None]
            * directions[None, :]
        )
        steering = np.exp(1j * phase)
        covariance = identity + rho * steering @ steering.conj().T
        sign, log_determinant = np.linalg.slogdet(covariance)
        if abs(sign - 1.0) > 1e-12:
            raise RuntimeError("the model covariance is not positive definite")
        quadratic = np.trace(np.linalg.solve(covariance, sample_covariance)).real
        values[index] = -snapshots * (log_determinant + quadratic)

    return values


def canonicalize_directions(
    candidates: np.ndarray,
    *,
    spacing: float,
    u_limit: float,
) -> np.ndarray:


    candidates = np.asarray(candidates, dtype=np.float64)
    phase_cycles = float(spacing) * float(u_limit) * candidates
    wrapped = phase_cycles - np.floor(phase_cycles + 0.5)
    return np.sort(2.0 * wrapped, axis=-1)
def physical_probes(sources: int, scale: float, seed: int) -> np.ndarray:

    rng = np.random.default_rng(seed)
    points = list(rng.uniform(-1, 1, (RANDOM_POINTS, sources)))
    bound = math.ceil(scale) + 2
    seams = [-1.0, 1.0] + [
        (2 * n + 1) / scale for n in range(-bound, bound + 1)
        if -1 < (2 * n + 1) / scale < 1
    ]
    for seam, direction, axis in itertools.product(seams, (-1, 1), range(sources)):
        value = seam + direction * DELTA
        if -1 < value < 1:
            point = np.linspace(-0.7, 0.7, sources)
            point[axis] = value
            points.append(point)
    edges = sorted({-1.0, 1.0} | {
        sign * scale - 2 * n
        for sign in (-1, 1) for n in range(-bound, bound + 1)
        if -1 < sign * scale - 2 * n < 1
    })
    intervals = list(zip(edges[:-1], edges[1:]))
    for cells in itertools.product(intervals, repeat=sources):
        q = np.array([lo + (hi - lo) * (i + 1) / (sources + 1)
                      for i, (lo, hi) in enumerate(cells)])
        points.append(q / scale)

    for edge, direction, axis in itertools.product(edges, (-1, 1), range(sources)):
        value = edge + direction * DELTA
        if -1 < value < 1:
            q = np.linspace(-0.6, 0.6, sources)
            q[axis] = value
            points.append(q / scale)
    for center in (-0.5, 0.0, 0.5):
        points.append((center + np.arange(sources) * DELTA) / scale)
    return np.unique(np.sort(np.asarray(points), axis=1), axis=0)


def ordered_rows(points: np.ndarray) -> np.ndarray:
    return points[np.lexsort(points.T[::-1])]


def independent_branches(q: np.ndarray, scale: float) -> np.ndarray:

    fibers = [
        [(value + 2 * n) / scale
         for n in range(math.ceil((-scale - value) / 2),
                        math.ceil((scale - value) / 2))]
        for value in q
    ]
    return ordered_rows(np.sort(np.array(list(itertools.product(*fibers))), axis=1))


def audit_observation(batch, spec, sources: int, cell: int, episode: int,
                      betas: list[float]) -> list[dict]:
    spacing = float(batch.spacing[0])
    scale = 2 * spacing * spec.u_limit
    seed = SEED + 10000 * sources + 100 * cell + episode
    physical = physical_probes(sources, scale, seed)
    q = quotient_coordinate(torch.from_numpy(physical), batch.spacing, spec).sort(1).values
    independent_q = canonicalize_directions(physical, spacing=spacing, u_limit=spec.u_limit)
    coordinate_error = float(np.max(np.abs(q.numpy() - independent_q)))
    produced = [[] for _ in q]
    for valid, points in branch_rows(q, scale):
        for index, point in zip(np.flatnonzero(valid.numpy()), points.numpy()):
            produced[index].append(point)
    actual = [ordered_rows(np.asarray(points)) for points in produced]
    expected = [independent_branches(point, scale) for point in independent_q]
    failures = []
    for index, (left, right) in enumerate(zip(actual, expected)):
        if left.shape != right.shape:
            failures.append({"probe": index, "kind": "branch_count"})
            continue
        coordinate_error = max(coordinate_error, float(np.max(np.abs(left - right))))
        if not np.all((left >= -1) & (left < 1)):
            failures.append({"probe": index, "kind": "physical_support"})
        if len(left) != len(np.unique(left, axis=0)):
            failures.append({"probe": index, "kind": "duplicate_branch"})
        if np.sum(np.max(np.abs(left - physical[index]), axis=1) < 1e-12) != 1:
            failures.append({"probe": index, "kind": "original_point_membership"})
    log_q = likelihood_in_chunks(canonical_batch(batch, spec), q, spec,
                                 chunk_size=4096).numpy()
    flat_expected = np.concatenate(expected)
    log_physical = direct_covariance_log_likelihood(
        batch.observations[0].numpy(), flat_expected, spacing=spacing,
        snr_db=float(batch.snr_db[0]), u_limit=spec.u_limit,
    )
    counts = np.array([len(points) for points in expected])
    expanded_log_q = np.repeat(log_q, counts)
    likelihood_error = float(np.max(np.abs(expanded_log_q - log_physical)))
    rows = []
    for beta in betas:
        base_density = math.factorial(sources) / 2**sources
        H = base_density * quotient_ratio(q, scale, beta).numpy()
        prior_reference = base_density * (1 + beta * np.prod(flat_expected, axis=1))
        H_expected = np.array([np.sum(p) for p in np.split(
            prior_reference, np.cumsum(counts)[:-1])]) / scale**sources
        row = {
            "sources": sources, "cell": cell, "episode": episode, "prior_beta": beta,
            "spacing": spacing, "snr_db": float(batch.snr_db[0]),
            "snapshots": int(batch.observations.shape[1]), "seed": seed,
            "physical_probes": len(physical), "branch_evaluations": len(flat_expected),
            "branch_counts_seen": sorted(set(counts.tolist())),
            "max_log_likelihood_error": likelihood_error,
            "max_coordinate_error": coordinate_error,
            "max_relative_H_error": float(np.max(np.abs(H / H_expected - 1))),
            "structural_failures": failures,
        }
        if any(left.shape != right.shape for left, right in zip(actual, expected)):
            row.update({key: None for key in (
                "max_log_density_error", "max_kappa_error", "max_kappa_sum_error")})
        else:
            prior_actual = [base_density * prior_density(torch.from_numpy(p), beta).numpy()
                            for p in actual]
            conditional = [p / p.sum() for p in prior_actual]
            kappa = np.concatenate(conditional)
            kappa_expected = prior_reference / np.repeat(H_expected * scale**sources, counts)
            log_restored = (expanded_log_q + np.repeat(np.log(H), counts)
                            + np.log(kappa) + sources * math.log(scale))
            log_direct = log_physical + np.log(prior_reference)
            row.update({
                "max_log_density_error": float(np.max(np.abs(log_restored - log_direct))),
                "max_kappa_error": float(np.max(np.abs(kappa - kappa_expected))),
                "max_kappa_sum_error": max(float(abs(p.sum() - 1)) for p in conditional),
            })
        row["passed"] = not failures and all(
            row[key] is not None and math.isfinite(row[key]) and row[key] <= tolerance
            for key, tolerance in TOLERANCES.items()
        )
        rows.append(row)
    return rows

def main():
    parser = argparse.ArgumentParser(description="Verify Table 1 on all 288 observations.")
    parser.add_argument("--output", type=Path, default=Path("verification.json"))
    args = parser.parse_args()
    torch.set_num_threads(1)
    rows = []
    for obs in range(288):
        spec, batch, sources, cell, episode = load_observation(obs)
        rows.extend(audit_observation(batch, spec, sources, cell, episode,
                                     [0.0, 0.7] if sources == 2 else [0.0]))
    groups = []
    for sources, beta in ((2, 0.0), (2, 0.7), (3, 0.0), (4, 0.0)):
        subset = [r for r in rows if r["sources"] == sources and r["prior_beta"] == beta]
        groups.append({
            "sources": sources, "prior_beta": beta, "observations": len(subset),
            "physical_probes": sum(r["physical_probes"] for r in subset),
            "branch_evaluations": sum(r["branch_evaluations"] for r in subset),
            **{key: max(r[key] for r in subset) for key in TOLERANCES},
            "failed_cases": sum(not r["passed"] for r in subset),
            "structural_failures": sum(len(r["structural_failures"]) for r in subset),
        })
    assert [g["observations"] for g in groups] == [192, 192, 48, 48]
    assert all(r["passed"] for r in rows)
    args.output.write_text(json.dumps({"groups": groups, "rows": rows}, indent=2) + "\n")
    print(json.dumps(groups, indent=2))


if __name__ == "__main__":
    main()
