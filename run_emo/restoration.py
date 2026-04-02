# Utility functions for restoring EMO results (compatible with pp4.py)

import os
import sys
sys.path.insert(0, os.path.abspath("./pymoo"))

import numpy as np
from copy import deepcopy
import struct
import base64


def load_all_FX(result_dir):
    """
    Purpose: Load all objective values stored in base64-encoded CSV.

    Each entry is decoded from base64 and unpacked as a double (float64).
    """

    file_path = os.path.join(result_dir, "all_f.csv")

    # Load base64-encoded strings
    encoded_array = np.loadtxt(file_path, dtype=str, delimiter=',')

    # Prepare output array
    decoded_objectives = np.empty_like(encoded_array, dtype=np.float64)

    for row_idx in range(encoded_array.shape[0]):
        for col_idx in range(encoded_array.shape[1]):
            decoded_objectives[row_idx, col_idx] = struct.unpack(
                '>d',
                base64.b64decode(encoded_array[row_idx, col_idx])
            )[0]

    return decoded_objectives


def load_id_arr(result_dir):
    """
    Purpose: Load index array mapping generations to population indices.
    """

    file_path = os.path.join(result_dir, "pop_f_id.csv")
    population_indices = np.loadtxt(file_path, delimiter=',', dtype="int64")

    return population_indices


def restore_pop_f(all_objectives, population_indices, generation):
    """
    Purpose: Restore parent population (P) at a given generation.
    """

    try:
        return all_objectives[population_indices[generation]]
    except IndexError:
        print("IndexError in restore_population_objectives")
        return


def restore_off_f(all_objectives, population_size, offspring_size, generation):
    """
    Purpose: Restore offspring population (Q) at a given generation.
    """

    start_idx = (generation - 1) * offspring_size + population_size
    end_idx = start_idx + offspring_size

    try:
        return all_objectives[start_idx:end_idx, :]
    except IndexError:
        print("IndexError in restore_offspring_objectives")
        return


if __name__ == '__main__':

    # Example usage for verification

    algorithm_name = "NSGA2"
    problem_name = "dtlz1"
    n_objectives = 2
    run_id = 0

    population_size = 100
    offspring_size = 1 if algorithm_name == "SMSEMOA" else 100

    generation = 893
    print("generation =", generation)

    result_dir = f"emo_results_tab/{algorithm_name}_mu{population_size}/{problem_name}_m{n_objectives}/runID{run_id}"

    all_objectives = load_all_FX(result_dir)
    population_indices = load_id_arr(result_dir)

    restored_population = restore_pop_f(
        all_objectives,
        population_indices,
        generation
    )

    # Ground truth (for validation)
    reference_file = f"emo_results/{algorithm_name}_mu{population_size}/{problem_name}_m{n_objectives}/runID{run_id}/pop_f_{generation}th_iter.csv"
    reference_population = np.loadtxt(reference_file, delimiter=',')

    # Compare restored vs original
    for restored, reference in zip(restored_population, reference_population):
        print(restored, reference)

    # Example (optional)
    # offspring = restore_off_f(
    #     all_objectives,
    #     population_size,
    #     offspring_size,
    #     generation
    # )