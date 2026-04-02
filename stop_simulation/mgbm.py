import os
import sys
import numpy as np
from copy import deepcopy

sys.path.append('../run_emo') 
from restoration import restore_pop_f, load_all_FX, load_id_arr

import click

def is_dominated_by(a, b):
    """
    Purpose: Check whether solution b dominates solution a.
    """
    return np.all(b <= a) and np.any(b < a)


def count_dominated_solutions(target_pop, reference_pop):
    """
    Purpose: Count how many solutions in target_pop are dominated
             by at least one solution in reference_pop.
    """
    dominated_count = 0

    for target in target_pop:
        if np.any([is_dominated_by(target, ref) for ref in reference_pop]):
            dominated_count += 1

    return dominated_count

def compute_mdr(current_population, previous_population):
    """
    Purpose: Compute MDR (Modified Dominance Ratio).

    Definition:
        MDR = (ratio of previous solutions dominated by current)
            - (ratio of current solutions dominated by previous)
    """

    prev_dominated_ratio = (
        count_dominated_solutions(previous_population, current_population)
        / len(previous_population)
    )

    curr_dominated_ratio = (
        count_dominated_solutions(current_population, previous_population)
        / len(current_population)
    )

    return prev_dominated_ratio - curr_dominated_ratio

@click.command()
@click.option('--alg_name', type=str, default="NSGA2")
@click.option('--problem_name', type=str, default="dtlz1")
@click.option('--n_obj', type=int, default=2)
def run(alg_name, problem_name, n_obj):
    """
    Purpose: Detect stopping generation using MDR + Kalman filter approach.
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
    observation_noise = 0.05  # R
    stopping_threshold = 0.12  # est_min

    output_dir = os.path.join(
        './fe_stop',
        f'{alg_name}_mu{population_size}',
        f'{problem_name}_m{n_obj}',
        'mgbm'
    )
    os.makedirs(output_dir, exist_ok=True)

    output_file = os.path.join(
        output_dir,
        f'R{observation_noise}_Imin{stopping_threshold}.csv'
    )

    stopping_fe_list = []

    for run_id in range(31):
        print("runID =", run_id)

        generation = 1

        # Kalman filter initialization
        posterior_estimate = 1.0
        posterior_covariance = observation_noise

        result_dir = os.path.join(
            '../run_emo/emo_results_tab',
            f'{alg_name}_mu{population_size}',
            f'{problem_name}_m{n_obj}',
            f'runID{run_id}'
        )

        all_objectives = load_all_FX(result_dir)
        id_array = load_id_arr(result_dir)

        previous_population = restore_pop_f(all_objectives, id_array, generation)

        for _ in range(2, len(id_array)):

            generation += 1
            current_population = restore_pop_f(all_objectives, id_array, generation)

            # Prediction step
            prior_estimate = posterior_estimate

            # Observation: improvement rate
            improvement_rate = compute_mdr(current_population, previous_population)

            # Update step (Kalman filter)
            prior_covariance = posterior_covariance
            kalman_gain = prior_covariance / (prior_covariance + observation_noise)

            posterior_estimate = (
                prior_estimate
                + kalman_gain * (improvement_rate - prior_estimate)
            )

            # Stopping conditions
            if generation == max_generations + 1:
                print("Reached max generations")
                stopping_fe_list.append(
                    population_size + offspring_size * (generation - 1)
                )
                break

            if posterior_estimate < stopping_threshold:
                print("Stop at generation:", generation)
                stopping_fe_list.append(
                    population_size + offspring_size * (generation - 1)
                )
                break

            # Update for next iteration
            previous_population = deepcopy(current_population)

            posterior_covariance = (
                observation_noise
                / (observation_noise + posterior_covariance)
                * posterior_covariance
            )

    print(stopping_fe_list)
    np.savetxt(output_file, np.array(stopping_fe_list), fmt="%d", delimiter=',')

if __name__ == '__main__':
    run()