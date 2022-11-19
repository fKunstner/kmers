import numpy as np
from solver_comparison.experiment import Experiment, ExperimentMonitor
from solver_comparison.problem.problem import Problem
from solver_comparison.solvers.initializer import Initializer
from solver_comparison.solvers.optimizer import GDLS

problems = [
    Problem(
        model_name="Logistic",
        filename="test5.fsa",
        K=8,
        N=1_000,
        L=14,
        alpha=0.1,
        beta=0.0,
    ),
    Problem(
        model_name="Simplex",
        filename="test5.fsa",
        K=8,
        N=1_000,
        L=14,
        alpha=0.1,
        beta=0.0,
    ),
]
optimizers = [GDLS(max=1.0, incr=1.1, max_iter=100)]


if __name__ == "__main__":

    np.seterr(all="raise")

    experiments = [
        Experiment(
            prob=prob,
            opt=opt,
            init=Initializer("zero"),
        )
        for prob in problems
        for opt in optimizers
    ]

    for exp in experiments:
        exp.run()
