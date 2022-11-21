import numpy as np
from solver_comparison.experiment import Experiment, ExperimentMonitor
from solver_comparison.plotting import make_individual_exp_plots
from solver_comparison.problem.problem import Problem
from solver_comparison.solvers.initializer import Initializer
from solver_comparison.solvers.optimizer import GDLS, ExpGrad

problems_simplex = [
    Problem(
        model_name="Simplex",
        filename="sampled_genome_0.001.fsa",
        K=8,
        N=1_000,
        L=14,
        alpha=0.1,
        beta=1.0,
    ),
]
optimizers_simplex = [ExpGrad()]


if __name__ == "__main__":

    np.seterr(all="raise")

    experiments_simplex = [
        Experiment(
            prob=prob,
            opt=opt,
            init=Initializer("simplex_uniform"),
        )
        for prob in problems_simplex
        for opt in optimizers_simplex
    ]
    experiments = experiments_simplex

    for exp in experiments:
        if not exp.has_already_run():
            exp.run()

    for exp in experiments:
        make_individual_exp_plots(exp)
