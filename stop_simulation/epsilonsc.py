import os
import sys
import numpy as np

from platypus import EpsilonBoxArchive, Solution, Problem

sys.path.append('../run_emo') 
from restoration import restore_pop_f, restore_off_f, load_all_FX, load_id_arr

import click

def add_population_to_archive(archive, population, n_obj):
    """
    Purpose: Add a population (objective vectors) to EpsilonBoxArchive.

    Note:
        Platypus requires Solution objects, so we wrap objective vectors
        into dummy Solution instances.
    """
    for obj_vector in population:
        problem = Problem(0, n_obj)
        solution = Solution(problem)
        solution.objectives[:] = obj_vector.tolist()
        archive.add(solution)

@click.command()
@click.option('--alg_name', type=str, default="NSGA2")
@click.option('--problem_name', type=str, default="dtlz1")
@click.option('--n_obj', type=int, default=2)
def run(alg_name, problem_name, n_obj):
    """
    Purpose: Detect stopping generation based on epsilon-archive stagnation.
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
    epsilon_value = 0.2
    stagnation_threshold = 10  # number of generations without improvement

    output_dir = os.path.join(
        './fe_stop',
        f'{alg_name}_mu{population_size}',
        f'{problem_name}_m{n_obj}',
        'epsilonsc'
    )
    os.makedirs(output_dir, exist_ok=True)

    output_file = os.path.join(
        output_dir,
        f'gen{stagnation_threshold}_ep{epsilon_value}.csv'
    )

    stopping_fe_list = []

    for run_id in range(31):

        result_dir = os.path.join(
            '../run_emo/emo_results_tab',
            f'{alg_name}_mu{population_size}',
            f'{problem_name}_m{n_obj}',
            f'runID{run_id}'
        )

        # Initialize epsilon archive
        archive = EpsilonBoxArchive([epsilon_value])

        all_objectives = load_all_FX(result_dir)
        id_array = load_id_arr(result_dir)

        # Add initial population
        initial_population = restore_pop_f(all_objectives, id_array, 1)
        add_population_to_archive(archive, initial_population, n_obj)

        stagnation_count = 0
        terminated_early = False

        # Iterate over generations
        for generation in range(1, len(id_array) - 1):

            offspring_population = restore_off_f(
                all_objectives,
                population_size,
                offspring_size,
                generation
            )

            previous_improvements = archive.improvements

            # Add offspring to archive
            add_population_to_archive(archive, offspring_population, n_obj)

            # Check improvement
            if archive.improvements > previous_improvements:
                stagnation_count = 0
            else:
                stagnation_count += 1

            # Stopping condition
            if stagnation_count >= stagnation_threshold:
                print(f"Terminate at generation {generation} (epsilon)")
                stopping_fe_list.append(
                    population_size + offspring_size * generation
                )
                terminated_early = True
                break

        # If not terminated early
        if not terminated_early:
            print("Terminate at max_generations")
            stopping_fe_list.append(
                population_size + offspring_size * max_generations
            )

    stopping_fe_array = np.array(stopping_fe_list)

    np.savetxt(output_file, stopping_fe_array, fmt="%d", delimiter=',')

    print(stopping_fe_array)
    print("mean =", np.mean(stopping_fe_array))
    print("-----------------------------------------------------")

if __name__ == '__main__':
    run()
                    