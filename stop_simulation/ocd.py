import statistics
import scipy.stats as sps
import numpy as np
import math
import pygmo as pg
import itertools
from copy import deepcopy

import os
import sys
sys.path.append('../run_emo') 
from restoration import restore_pop_f, load_all_FX, load_id_arr

import click

EPS = 1e-12

def calc_hv(point_set, hv_ref_point):
    del_mask = np.full(len(point_set), True)
    for i, p in enumerate(point_set):
        del_mask[i] = pg.pareto_dominance(p, hv_ref_point)
    point_set = point_set[del_mask]

    if len(point_set) != 0:
        hv = pg.hypervolume(point_set)
        hv_value = hv.compute(hv_ref_point)
    else:
        hv_value = 0.0

    return hv_value

def HyperVolume(prev_pf_norm, curr_pf_norm, weights, ideal_point, nadir_point):
    """
    Purpose: Compute the difference in hypervolume between two Pareto fronts.

    Input:
        prev_pf_norm: normalized PF of previous generation
        curr_pf_norm: normalized PF of current generation

    Output:
        Difference in HV (prev - current)
    """

    if len(prev_pf_norm) == 0:
        prev_hv = 0.0
    else:
        prev_pf = np.array(prev_pf_norm, dtype=float)
        n_obj = prev_pf.shape[1]
        reference_point = [2.1] * n_obj
        prev_hv = calc_hv(prev_pf, reference_point)

    if len(curr_pf_norm) == 0:
        curr_hv = 0.0
    else:
        curr_pf = np.array(curr_pf_norm, dtype=float)
        n_obj = curr_pf.shape[1]
        reference_point = [2.1] * n_obj
        curr_hv = calc_hv(curr_pf, reference_point)

    return float(prev_hv - curr_hv)

def epsilon(prev_pf_norm, curr_pf_norm, weights, ideal_point, nadir_point):
    """
    Purpose: Compute additive epsilon indicator.

    For each solution in current PF, find the minimum epsilon
    required to dominate it using previous PF.
    """

    prev_pf = np.array(prev_pf_norm, dtype=float)
    curr_pf = np.array(curr_pf_norm, dtype=float)

    n_obj = prev_pf.shape[1]
    max_epsilon = -1e10

    for ref_sol in curr_pf:
        min_epsilon = 1e10

        for pf_sol in prev_pf:
            max_diff = -1e10

            for obj_idx in range(n_obj):
                diff = pf_sol[obj_idx] - ref_sol[obj_idx]
                max_diff = max(max_diff, diff)

            min_epsilon = min(min_epsilon, max_diff)

        max_epsilon = max(max_epsilon, min_epsilon)

    return float(max_epsilon)

def R2(prev_pf_norm, curr_pf_norm, weights, ideal_point, nadir_point):
    """
    Purpose: Compute R2 indicator difference using Tchebycheff approach.
    """

    prev_pf = np.array(prev_pf_norm, dtype=float)
    curr_pf = np.array(curr_pf_norm, dtype=float)
    weights = np.array(weights, dtype=float)
    ideal_point = np.array(ideal_point, dtype=float)

    prev_diff = prev_pf - ideal_point
    curr_diff = curr_pf - ideal_point

    prev_values = []
    curr_values = []

    for w in weights:
        prev_tchebycheff = np.max(w * prev_diff, axis=1)
        curr_tchebycheff = np.max(w * curr_diff, axis=1)

        prev_values.append(np.min(prev_tchebycheff))
        curr_values.append(np.min(curr_tchebycheff))

    return float(np.mean(prev_values) - np.mean(curr_values))

# Statistical helpers (Chi2 and Reg) — keep your existing implementations but ensure input PI arrays are proper floats
def Chi2(PI, VarLimit):
    N = len(PI) - 1
    if N <= 0:
        return 1.0
    Chi = (statistics.variance(PI) * N) / VarLimit
    p = sps.chi2.cdf(Chi, N)
    return float(p)

def Reg(PI):
    # same as before but defensive about shapes
    n = len(PI)
    L = len(PI[0])
    N = n * L - 1
    PI_std = []
    for j in range(n):
        PI_std.append(sps.zscore(PI[j]) if np.std(PI[j]) > 0 else np.zeros_like(PI[j]))
    Y = np.concatenate(PI_std)
    X = np.tile(np.arange(1, L + 1), n)
    Xs = sps.zscore(X) if np.std(X) > 0 else X
    Xs = Xs.reshape(1, -1)
    Y = Y.reshape(1, -1)

    XX_inv = 1.0 / (Xs @ Xs.T)[0][0] if (Xs @ Xs.T)[0][0] != 0 else 0.0
    beta = (XX_inv * Xs @ Y.T)[0][0]
    eps = Y - Xs * beta
    s2 = (eps @ eps.T)[0][0] / max(N, 1)
    t = beta / math.sqrt(max(s2 * XX_inv, EPS))
    p = 2 * min(sps.t.cdf(t, N), 1 - sps.t.cdf(t, N))
    return float(p)

def normalize_pf(pf, lb, ub):
    pf = np.array(pf, dtype=float)
    return (pf - lb) / (ub - lb + 1e-12) + 1

def generate_weights(n_weights, m):
    def comb(n,r):
        from math import comb as c
        return c(n,r)
    
    H = 1
    while comb(H+m-1, m-1) < n_weights:
        H += 1
    
    weights = []
    for c in itertools.combinations_with_replacement(range(H + m - 1), m - 1):
        c = (-1,) + c + (H + m - 1,)
        w = np.diff(c) - 1 
        w = w / H 
        weights.append(w)
    weights = np.array(weights)

    if len(weights) > n_weights:
        idx = np.random.choice(len(weights), n_weights, replace=False)
        weights = weights[idx]
    
    return weights
def detect_convergence(
    variance_limit,
    window_size,
    alpha,
    max_generations,
    performance_indicators,
    population_history,
    weights,
    n_objectives
):
    """
    Purpose: Detect convergence generation using OCD criteria.

    Returns:
        Convergence generation index
    """

    generation = 1
    n_metrics = len(performance_indicators)

    chi2_pvalues = [[1] * (max_generations + 2) for _ in range(n_metrics)]
    regression_pvalues = [0] * (max_generations + 2)

    lower_bound = np.full(n_objectives, 1e10)
    upper_bound = np.full(n_objectives, -1e10)

    normalized_history = deepcopy(population_history)

    while True:
        generation += 1

        # Update ideal and nadir points
        current_pf = np.array(population_history[generation])
        lower_bound = np.minimum(lower_bound, current_pf.min(axis=0))
        upper_bound = np.maximum(upper_bound, current_pf.max(axis=0))

        if generation > window_size:

            # Normalize PFs in sliding window
            for past_gen in range(generation - window_size, generation + 1):
                normalized_history[past_gen] = normalize_pf(
                    population_history[past_gen],
                    lower_bound,
                    upper_bound
                )

            # Compute performance indicator values
            pi_values = []

            for metric_idx in range(n_metrics):
                metric_values = []

                for past_gen in range(generation - window_size, generation):
                    metric_values.append(
                        performance_indicators[metric_idx](
                            normalized_history[past_gen],
                            normalized_history[generation],
                            weights,
                            np.full(n_objectives, 1),
                            np.full(n_objectives, 2)
                        )
                    )

                pi_values.append(metric_values)
                chi2_pvalues[metric_idx][generation] = Chi2(metric_values, variance_limit)

            # Regression test
            regression_pvalues[generation] = Reg(pi_values)

        # Stopping conditions
        cond_variance = (
            all(chi2_pvalues[j][generation] <= alpha / n_metrics for j in range(n_metrics)) and
            all(chi2_pvalues[j][generation - 1] <= alpha / n_metrics for j in range(n_metrics))
        )

        cond_trend = (
            regression_pvalues[generation] > alpha and
            regression_pvalues[generation - 1] > alpha
        )

        cond_max_iter = (generation == max_generations + 1)

        if cond_variance or cond_trend or cond_max_iter:
            break

    return generation - 1

@click.command()
@click.option('--alg_name', type=str, default="NSGA2")
@click.option('--problem_name', type=str, default="dtlz1")
@click.option('--n_obj', type=int, default=2)
def run(alg_name, problem_name, n_obj):
    pop_size = 100
    if alg_name == "SMSEMOA":
        children_size = 1
    else:
        children_size = pop_size

    max_fevals = 100000

    if alg_name == "SMSEMOA":
        max_iters = int(max_fevals - pop_size)
    else:
        max_iters = int(max_fevals / pop_size) - 1
    
    VarLimit = 0.001**2
    nPreGen = 10
    alpha = 0.05
    PI = [HyperVolume, epsilon, R2]

    Weight = generate_weights(pop_size, n_obj)
    fe_stop_dir_path = os.path.join('./fe_stop', alg_name+f'_mu{pop_size}', problem_name+f'_m{n_obj}', 'ocd')
    os.makedirs(fe_stop_dir_path, exist_ok=True)
    fe_stop_file_path = os.path.join(fe_stop_dir_path, f'gen{nPreGen}_var{VarLimit}.csv')
    stop_fe_list = []
    for run_id in range(31):
        print("runID = ", run_id)
        res_dir_path = os.path.join('../run_emo/emo_results_tab', alg_name+f'_mu{pop_size}', problem_name+f'_m{n_obj}', f'runID{run_id}')

        PF_EMOA = []
        all_FX = load_all_FX(res_dir_path)
        id_arr = load_id_arr(res_dir_path)
        for iter_count in range(len(id_arr)):
            PF_EMOA.append(restore_pop_f(all_FX, id_arr, iter_count))
        iter_stop = detect_convergence(VarLimit, nPreGen, alpha, max_iters, PI, PF_EMOA, Weight, n_obj)
        stop_fe_list.append(pop_size + children_size * iter_stop)

    np.savetxt(fe_stop_file_path, stop_fe_list, fmt="%d", delimiter=',')
    print(fe_stop_file_path)

if __name__ == '__main__':
    run()
                        
                