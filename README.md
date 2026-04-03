# emo_stopping_criteria
This Python code is to implement five EMO stopping criteria discussed in the following paper:
> Kenji Kitamura, Ryoji Tanabe, Benchmarking Stopping Criteria for Evolutionary Multi-objective Optimization, GECCO 2026

# Requirements
This code at least require Python 3 and numpy. Platypus library is also required to run εSC (https://github.com/Project-Platypus/Platypus).

# Usage
The following command runs OCD for NSGA-II on DTLZ1 with 2 objectives :
```bash
$ python ocd.py --alg_name NSGA2 --problem_name dtlz1 --n_obj 2
```
Other stopping criteria can be executed using similar commands.
