import subprocess

if __name__ == '__main__':
    for alg_name in ["NSGA2"]:
        for problem_name in ["dtlz1"]:
            for n_obj in [2]:
                for sc in ["ocd"]:#, "mgbm"]:#, "esc", "epsilonsc", "isc"]:

                    envs = [
                        f"ALG_NAME={alg_name}",
                        f"PROBLEM_NAME={problem_name}",
                        f"N_OBJ={n_obj}",
                    ]

                    subprocess.run([
                        "qsub",
                        "-l", "walltime=48:00:00",
                        "-v", ",".join(envs),
                        f"job_{sc}.sh",
                    ])
