import argparse
import hashlib
import json
import math
from pathlib import Path
import platform
import time
import numpy as np
import pandas as pd
import ot
from scipy.spatial.distance import cdist
from scipy.special import softmax
import torch
from model import ROOT, METHODS, load_observation, sobol_nodes, evaluate_nodes

BUDGETS = (1024, 4096, 16384, 65536, 262144)
SCRAMBLES = tuple(range(931101, 931133))
REFERENCE_SEEDS = ((932101, 932102, 932103, 932104), (933101, 933102, 933103, 933104))
REFERENCE_POINTS = 1048576
def rng_for(*keys):
    return np.random.default_rng(np.random.SeedSequence([937101, *map(int, keys)]))


def parent_distribution(x, logw, method, scale):

    if method == METHODS[0]:
        return softmax(logw), None, None
    lower = np.ceil((-scale - x) / 2).astype(np.int64)
    upper = np.ceil((scale - x) / 2).astype(np.int64) - 1
    counts = upper - lower + 1
    assert (counts > 0).all()
    w = softmax(logw + np.log(counts).sum(axis=1))
    return w, lower, counts


def sample_measure(x, w, lower, counts, scale, size, rng):
    cdf = np.cumsum(w)
    cdf /= cdf[-1]
    ids = np.searchsorted(cdf, rng.random(size), side='right')
    points = x[ids].copy()
    if lower is not None:
        shifts = lower[ids] + np.floor(rng.random(points.shape)*counts[ids]).astype(int)
        points = np.sort((points + 2*shifts)/scale, axis=1)
    assert np.isfinite(points).all() and (np.abs(points) <= 1).all()
    return points


def distance(x, y, metric='cityblock'):

    start = time.perf_counter()
    x, ac = np.unique(x, axis=0, return_counts=True)
    y, bc = np.unique(y, axis=0, return_counts=True)
    cost = cdist(x, y, metric=metric)
    if metric == 'cityblock':
        cost /= x.shape[1]
    a, b = ac/ac.sum(), bc/bc.sum()
    units = math.lcm(int(ac.sum()), int(bc.sum()))
    integer_a = (ac*(units//int(ac.sum()))).astype(np.float64)
    integer_b = (bc*(units//int(bc.sum()))).astype(np.float64)


    scale_cost = 1e12
    solver_cost = np.rint(cost*scale_cost)
    plan, log = ot.emd(integer_a, integer_b, solver_cost, numItermax=10000000, log=True, numThreads=1)
    assert log['warning'] is None, log
    plan /= units
    primal = float(np.sum(plan*cost))
    u, v = log['u']/scale_cost, log['v']/scale_cost
    violation = float(np.maximum(u[:, None]+v[None, :]-cost, 0).max())
    dual = float(a@u + b@v) - violation - 2e-12
    residual = max(float(np.abs(plan.sum(1)-a).sum()), float(np.abs(plan.sum(0)-b).sum()))
    gap = abs(primal-dual)
    assert residual <= 1e-9 and violation <= 1e-9 and gap <= 1e-9
    return {'w1': primal, 'lower': dual, 'gap': gap, 'dual_violation': violation,
            'mass_residual': residual, 'unique_candidate_atoms': len(x),
            'unique_reference_atoms': len(y), 'seconds': time.perf_counter()-start}
@torch.no_grad()
def draw(batch, spec, k, method, n, obs):
    nodes = sobol_nodes(k, 931101, n)
    logw = evaluate_nodes(batch, spec, nodes, method)[0].numpy()
    scale = 2 * float(batch.spacing[0]) * spec.u_limit
    x = nodes.numpy()
    w, lower, counts = parent_distribution(x, logw, method, scale)
    return sample_measure(x, w, lower, counts, scale, 100,
                          rng_for(4, obs, 931101, METHODS.index(method), n, 100, 0))

def accuracy(observations, output):
    rows, baselines = [], []
    for obs in observations:
        spec, batch, k, _, _ = load_observation(obs)
        scale = 2 * float(batch.spacing[0]) * spec.u_limit
        references = {}
        for pool, seeds in enumerate(REFERENCE_SEEDS):
            nodes = [sobol_nodes(k, seed, REFERENCE_POINTS) for seed in seeds]
            logits = [evaluate_nodes(batch, spec, x, METHODS[0])[0].numpy() for x in nodes]
            x, logw = np.concatenate([t.numpy() for t in nodes]), np.concatenate(logits)
            del nodes, logits
            weights = softmax(logw)
            references[pool] = sample_measure(x, weights, None, None, scale, 10000,
                                              rng_for(1, obs, pool))
            for rep in range(32):
                samples = sample_measure(x, weights, None, None, scale, 100,
                                         rng_for(2, obs, pool, rep))
                baselines.append({'obs': obs, 'k': k, 'pool': pool, 'repeat': rep,
                                  **distance(samples, references[0])})
            del x, logw, weights
        for seed in SCRAMBLES:
            nodes = sobol_nodes(k, seed, max(BUDGETS))
            x = nodes.numpy()
            for mi, method in enumerate(METHODS):
                logw = evaluate_nodes(batch, spec, nodes, method)[0].numpy()
                for n in BUDGETS:
                    weights, lower, counts = parent_distribution(x[:n], logw[:n], method, scale)
                    samples = sample_measure(x[:n], weights, lower, counts, scale, 100,
                                             rng_for(4, obs, seed, mi, n, 100, 0))
                    for pool in (0, 1):
                        rows.append({'obs': obs, 'k': k, 'method': method, 'seed': seed,
                                     'n': n, 'reference_pool': pool,
                                     **distance(samples, references[pool])})
        print(f'Observation {obs}: complete', flush=True)
    for name, values in (('distances', rows), ('reference_baselines', baselines)):
        frame = pd.DataFrame(values).drop(columns=['seconds','unique_candidate_atoms','unique_reference_atoms'])
        frame.to_csv(output / f'{name}.csv.gz', index=False,
                     compression={'method':'gzip', 'mtime':0})
    summarize(output, output)


def summarize(source, output):
    df = pd.read_csv(source / 'distances.csv.gz', float_precision='round_trip')
    baseline = pd.read_csv(source / 'reference_baselines.csv.gz', float_precision='round_trip')
    assert not df.duplicated(['obs','method','seed','n','reference_pool']).any()
    assert (df.groupby(['obs','n','method','reference_pool']).size() == 32).all()
    assert tuple(sorted(df.n.unique())) == BUDGETS
    assert tuple(sorted(df.seed.unique())) == SCRAMBLES
    assert not baseline.duplicated(['obs','pool','repeat']).any()
    assert (baseline.groupby(['obs','pool']).size() == 32).all()
    for field in ('gap','dual_violation','mass_residual'):
        assert max(df[field].max(), baseline[field].max()) <= 1e-9
    assert np.isfinite(df.w1).all() and (df.w1 >= 0).all()
    med = df.groupby(['obs','k','n','reference_pool','method']).w1.median().unstack('method').reset_index()
    med['ratio'] = med[METHODS[0]] / med[METHODS[1]]
    floor = baseline[baseline.pool == 0].groupby('obs').w1.median()
    med['floor'] = med.obs.map(floor)
    med.to_csv(output / 'per_observation.csv', index=False)
    sensitivity = baseline.groupby(['obs','k','pool']).w1.median().unstack('pool').reset_index()
    sensitivity = sensitivity.rename(columns={0:'same_pool_median', 1:'independent_pool_median'})
    sensitivity['ratio'] = sensitivity.independent_pool_median / sensitivity.same_pool_median
    sensitivity.to_csv(output / 'reference_sensitivity.csv', index=False)
    costs = []
    for obs, group in med[med.reference_pool == 0].groupby('obs'):
        group = group.sort_values('n')
        for multiplier in (1.25, 1.5, 2.0):
            threshold = float(group.floor.iloc[0]) * multiplier
            row = {'obs':int(obs), 'k':int(group.k.iloc[0]),
                   'multiplier':multiplier, 'threshold':threshold}
            for method in METHODS:
                ok = group[method].to_numpy() <= threshold
                sustained = np.logical_and.accumulate(ok[::-1])[::-1]
                row[method] = int(group.n.to_numpy()[sustained][0]) if sustained.any() else None
            costs.append(row)
    costs = pd.DataFrame(costs)
    costs.to_csv(output / 'common_accuracy.csv', index=False)
    groups = []
    for (k,n,pool), group in med.groupby(['k','n','reference_pool']):
        groups.append({'k':int(k), 'n':int(n), 'reference_pool':int(pool),
                       'observations':len(group),
                       'physical_median':float(group[METHODS[0]].median()),
                       'quotient_median':float(group[METHODS[1]].median()),
                       'paired_ratio_median':float(group.ratio.median()),
                       'quotient_point_wins':int((group.ratio > 1).sum()),
                       'floor_median':float(group.floor.median())})
    cost_groups = []
    for k, group in costs[costs.multiplier == 1.5].groupby('k'):
        row = {'k':int(k), 'observations':len(group)}
        for method in METHODS:
            row[method] = {'mean_budget':float(group[method].mean()),
                           'reached_at_1024':int((group[method] <= 1024).sum()),
                           'reached':int(group[method].notna().sum())}
        row['mean_budget_ratio'] = float(group[METHODS[0]].mean() / group[METHODS[1]].mean())
        cost_groups.append(row)
    result = {'observations':int(df.obs.nunique()), 'scores':len(df),
              'accuracy':groups, 'cost':cost_groups,
              'reference':{'points_per_rule':REFERENCE_POINTS, 'rules_per_pool':4,
                           'seeds':REFERENCE_SEEDS, 'reference_draws':10000, 'output_draws':100,
                           'scope':'Numerical reference sensitivity, not a continuum accuracy certificate; candidate draws are fixed across reference pools.',
                           'limitations':'Unresolved three-source reference uncertainty remains. Finite-reference sample comparisons do not certify convergence to the true posterior.'}}
    timing_path = source / 'timing.csv'
    if timing_path.exists():
        timing = pd.read_csv(timing_path, float_precision='round_trip')
        assert (timing.groupby(['obs','method']).size() == 5).all()
        times = timing.groupby(['obs','k','method','n']).ms.median().reset_index()
        result['timing'] = [{'k':int(k),'method':method,'mean_ms':float(g.ms.mean())}
                            for (k,method),g in times.groupby(['k','method'])]
    (output / 'summary.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({'observations':result['observations'], 'cost':cost_groups,
                      'timing':result.get('timing')}, indent=2))


def timing(source, output):
    costs = pd.read_csv(source / 'common_accuracy.csv')
    costs = costs[costs.multiplier == 1.5].set_index('obs')
    assert len(costs) == 240 and not costs[list(METHODS)].isna().any().any()
    rows = []
    for obs in range(240):
        spec, batch, k, _, _ = load_observation(obs)
        expected = {}
        for method in METHODS:
            points = draw(batch, spec, k, method, int(costs.loc[obs,method]), obs)
            assert points.shape == (100,k) and points.dtype == np.float64
            expected[method] = hashlib.sha256(points.tobytes()).hexdigest()
        schedule = [(method,rep) for method in METHODS for rep in range(5)]
        rng_for(940111, obs).shuffle(schedule)
        for method,rep in schedule:
            n = int(costs.loc[obs,method])
            start = time.perf_counter_ns()
            points = draw(batch, spec, k, method, n, obs)
            ms = (time.perf_counter_ns() - start) / 1e6
            assert hashlib.sha256(points.tobytes()).hexdigest() == expected[method]
            rows.append({'obs':obs,'k':k,'method':method,'n':n,'repeat':rep,'ms':ms})
    pd.DataFrame(rows).to_csv(output / 'timing.csv', index=False)
    environment = {'platform':platform.platform(), 'machine':platform.machine(),
                   'python':platform.python_version(), 'torch':str(torch.__version__),
                   'numpy':np.__version__, 'threads':1, 'repeats':5,
                   'scope':'100 draws at preselected budgets; includes Sobol initialization, likelihood, normalization and physical sampling; excludes loading, warmup, references and budget selection.'}
    (output / 'timing_environment.json').write_text(json.dumps(environment,indent=2)+'\n')
    print(pd.DataFrame(rows).groupby(['obs','k','method']).ms.median().groupby(['k','method']).mean())


def main():
    parser = argparse.ArgumentParser(
        description='Reproduce the RQMC accuracy comparison and common-accuracy timing.',
        epilog='Install: pip install -r requirements.txt. Verify: python verify.py. '
               'Saved results: python experiments.py summarize. Figure: python plot.py. '
               'Full accuracy: python experiments.py accuracy. '
               'Timing: python experiments.py timing. '
               'Accuracy recomputes two four-rule references and 32 scrambles per observation; '
               'it can take hours on CPU. Timing depends on hardware and system load.')
    parser.add_argument('action', choices=('accuracy','summarize','timing'))
    parser.add_argument('--input', type=Path, default=ROOT / 'results')
    parser.add_argument('--output', type=Path, default=ROOT / 'output')
    parser.add_argument('--observations', type=int, nargs='+', default=list(range(240)),
                        help='Observation indices for accuracy; default: all 240.')
    args = parser.parse_args()
    assert all(0 <= obs < 240 for obs in args.observations)
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    args.output.mkdir(parents=True, exist_ok=True)
    if args.action == 'accuracy':
        accuracy(args.observations, args.output)
    elif args.action == 'summarize':
        summarize(args.input, args.output)
    else:
        timing(args.input, args.output)


if __name__ == '__main__':
    main()
