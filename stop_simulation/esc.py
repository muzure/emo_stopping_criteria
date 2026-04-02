import os
import sys
import numpy as np
import math
from decimal import Decimal, ROUND_HALF_UP, ROUND_HALF_EVEN
from copy import deepcopy

sys.path.append('../run_emo') 
from restoration import restore_pop_f, load_all_FX, load_id_arr

import click

def compute_cell_id(solution, lower_bound, upper_bound, n_bins):
    """
    Purpose: Map a solution to a discretized grid cell ID.
    """

    normalized = (solution - lower_bound) / (upper_bound - lower_bound + 1e-12)

    # Handle NaN (edge case when ub == lb)
    if np.any(np.isnan(normalized)):
        normalized = np.zeros_like(solution)

    cell_id = 0
    for dim_idx, value in enumerate(normalized):
        bin_index = int(value * n_bins)
        cell_id += int(bin_index * (n_bins ** dim_idx))

    return cell_id

def get_lower_upper_bound(point_set1, point_set2):
    point_set = np.concatenate([point_set1, point_set2])
    lower_bound = point_set.min(axis=0)
    upper_bound = point_set.max(axis=0)

    return lower_bound, upper_bound

def build_multi_histogram(prev_pop, curr_pop, n_bins):
    """
    Purpose: Construct histogram counts for two populations.

    Returns:
        shared_cells: cells appearing in prev_pop
        unique_curr_cells: cells appearing only in curr_pop
        prev_counts: counts of prev_pop in shared_cells
        curr_counts_shared: counts of curr_pop in shared_cells
        curr_counts_unique: counts of curr_pop in unique_curr_cells
    """

    shared_cells = []
    unique_curr_cells = []

    prev_counts = []
    curr_counts_shared = []
    curr_counts_unique = []

    lower_bound, upper_bound = get_lower_upper_bound(prev_pop, curr_pop)

    # Count previous population
    for sol in prev_pop:
        cell_id = compute_cell_id(sol, lower_bound, upper_bound, n_bins)

        if cell_id in shared_cells:
            idx = shared_cells.index(cell_id)
            prev_counts[idx] += 1
        else:
            shared_cells.append(cell_id)
            prev_counts.append(1)
            curr_counts_shared.append(0)

    # Count current population
    for sol in curr_pop:
        cell_id = compute_cell_id(sol, lower_bound, upper_bound, n_bins)

        if cell_id in shared_cells:
            idx = shared_cells.index(cell_id)
            curr_counts_shared[idx] += 1
        elif cell_id in unique_curr_cells:
            idx = unique_curr_cells.index(cell_id)
            curr_counts_unique[idx] += 1
        else:
            unique_curr_cells.append(cell_id)
            curr_counts_unique.append(1)

    return shared_cells, unique_curr_cells, prev_counts, curr_counts_shared, curr_counts_unique

def all_values_equal(values):
    """
    Purpose: Check whether all elements in the list are equal.
    """
    return len(set(values)) <= 1

@click.command()
@click.option('--alg_name', type=str, default="NSGA2")
@click.option('--problem_name', type=str, default="dtlz1")
@click.option('--n_obj', type=int, default=2)
def run(alg_name, problem_name, n_obj):
    """
    Purpose: Detect stopping generation using entropy-based stability.
    """

    population_size = 100

    if alg_name == "SMSEMOA":
        offspring_size = 1
    else:
        offspring_size = population_size

    max_function_evals = 100000

    if alg_name == "SMSEMOA":
        max_generations = int(max_function_evals - population_size)
    else:
        max_generations = int(max_function_evals / population_size) - 1

    # Hyperparameters
    rounding_precision = '0.001'
    window_size = 20
    n_bins = 10

    output_dir = os.path.join(
        './fe_stop',
        f'{alg_name}_mu{population_size}',
        f'{problem_name}_m{n_obj}',
        'esc'
    )
    os.makedirs(output_dir, exist_ok=True)

    output_file = os.path.join(
        output_dir,
        f'gen{window_size}_dcm{rounding_precision}_bins{n_bins}.csv'
    )

    stopping_fe_list = []

    for run_id in range(31):
        print("run_id =", run_id)

        generation = 1
        condition_mean = False
        condition_variance = False

        cumulative_measure = 0.0
        cumulative_squared_measure = 0.0

        mean_history = [0] * (window_size - 1)
        variance_history = [0] * (window_size - 1)

        result_dir = os.path.join(
            '../run_emo/emo_results_tab',
            f'{alg_name}_mu{population_size}',
            f'{problem_name}_m{n_obj}',
            f'runID{run_id}'
        )

        all_objectives = load_all_FX(result_dir)
        id_array = load_id_arr(result_dir)

        prev_population = restore_pop_f(all_objectives, id_array, generation)
        generation += 1

        for _ in range(2, len(id_array)):

            current_population = restore_pop_f(all_objectives, id_array, generation)

            (
                shared_cells,
                unique_curr_cells,
                prev_counts,
                curr_counts_shared,
                curr_counts_unique
            ) = build_multi_histogram(prev_population, current_population, n_bins)

            # Entropy-like measure
            measure = 0.0

            for idx, cell in enumerate(shared_cells):
                p_prob = prev_counts[idx] / len(prev_population)
                q_prob = curr_counts_shared[idx] / len(current_population)

                if q_prob > 0:
                    measure -= (
                        (p_prob / 2) * math.log(p_prob / q_prob) +
                        (q_prob / 2) * math.log(p_prob / q_prob)
                    )
                else:
                    measure -= p_prob * math.log(p_prob)

            for idx, cell in enumerate(unique_curr_cells):
                q_prob = curr_counts_unique[idx] / len(current_population)
                measure -= q_prob * math.log(q_prob)

            # Update statistics
            cumulative_measure += measure
            cumulative_squared_measure += measure ** 2

            mean_value = cumulative_measure / (generation + 1)
            variance_value = (
                cumulative_squared_measure / (generation + 1)
                - mean_value ** 2
            )

            # Rounding (stability detection)
            mean_rounded = float(
                Decimal(str(mean_value)).quantize(
                    Decimal(rounding_precision), ROUND_HALF_UP
                )
            )

            variance_rounded = float(
                Decimal(str(variance_value)).quantize(
                    Decimal(rounding_precision), ROUND_HALF_UP
                )
            )

            mean_history.append(mean_rounded)
            variance_history.append(variance_rounded)

            if generation > window_size:
                if all_values_equal(mean_history):
                    condition_mean = True
                if all_values_equal(variance_history):
                    condition_variance = True

            # Stopping condition
            if condition_mean and condition_variance:
                print(f"Terminate at generation {generation - 1} (entropy)")
                stopping_fe_list.append(
                    population_size + offspring_size * (generation - 1)
                )
                break

            # Slide window
            del mean_history[0]
            del variance_history[0]

            generation += 1

            if generation > max_generations + 1:
                print("Reached max generations")
                stopping_fe_list.append(
                    population_size + offspring_size * max_generations
                )
                break

            # Reset flags and update population
            condition_mean = False
            condition_variance = False
            prev_population = deepcopy(current_population)

    print("mean stopping FE =", sum(stopping_fe_list) / len(stopping_fe_list))
    print("---------------------------------------------------------")

    np.savetxt(output_file, stopping_fe_list, fmt="%d", delimiter=',')
    print(output_file)  

if __name__ == '__main__':
    run()

                    

