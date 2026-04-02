# Stop when best-so-far hypervolume (HV) stagnates for T generations

import os
import sys
import numpy as np

import click


@click.command()
@click.option('--alg_name', type=str, default="NSGA2")
@click.option('--problem_name', type=str, default="dtlz1")
@click.option('--n_obj', type=int, default=2)
def run(alg_name, problem_name, n_obj):
    """
    Purpose: Detect stopping generation based on stagnation of best-so-far HV.
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

    # Hyperparameter: number of consecutive stagnation generations
    stagnation_threshold = 20

    output_dir = os.path.join(
        './fe_stop',
        f'{alg_name}_mu{population_size}',
        f'{problem_name}_m{n_obj}',
        'isc'
    )
    os.makedirs(output_dir, exist_ok=True)

    output_file = os.path.join(
        output_dir,
        f'gen{stagnation_threshold}.csv'
    )

    stopping_fe_list = []

    for run_id in range(31):
        print("runID =", run_id)

        hv_file_path = os.path.join(
            '../run_emo/hv_values_bsf',
            f'{alg_name}_mu{population_size}',
            f'{problem_name}_m{n_obj}',
            f'runID{run_id}.csv'
        )

        hv_values = np.loadtxt(hv_file_path)

        stagnation_count = 0
        stop_generation = -1

        for generation in range(2, len(hv_values)):

            # Skip invalid HV values (initial phase)
            if hv_values[generation - 1] == 0.0:
                continue

            # Check stagnation (no improvement)
            if hv_values[generation - 1] == hv_values[generation]:
                stagnation_count += 1

                if stagnation_count == stagnation_threshold:
                    stop_generation = generation - 1
                    break
            else:
                stagnation_count = 0

        # If no stagnation detected, stop at max_generations
        if stop_generation == -1:
            stop_generation = max_generations

        stopping_fe_list.append(
            population_size + offspring_size * stop_generation
        )

    print(stopping_fe_list)

    np.savetxt(output_file, np.array(stopping_fe_list), fmt="%d", delimiter=',')


if __name__ == '__main__':
    run()