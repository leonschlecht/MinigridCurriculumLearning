from pymoo.core.evaluator import Evaluator
from pymoo.core.problem import Problem
import numpy as np

from curricula import RollingHorizonEvolutionaryAlgorithm
from utils import ENV_NAMES


class CurriculumProblem(Problem):
    def __init__(self, curricula: list, n_obj, n_ieq_constr, xu, evolCurric: RollingHorizonEvolutionaryAlgorithm):
        assert len(curricula) > 0
        assert evolCurric is not None
        self.evolCurric = evolCurric
        n_var = len(curricula[0])
        # TODO maybe try to avoid homogenous curricula with ieq constraints (?)
        xl = np.zeros(n_var, dtype=int)
        xu = np.full(n_var, xu, dtype=int)
        super().__init__(n_var=n_var,
                         n_obj=n_obj,  # maximizing the overall reward = 1 objective
                         n_ieq_constr=n_ieq_constr,  # = 0 ?
                         xl=xl,
                         xu=xu)
        self.curricula = curricula
        self.N = 0

        # F: what we want to maximize: ---> pymoo minimizes, so it should be -reward
        # G:# Inequality constraint;
        # H is EQ constraint: maybe we can experiment with the length of each curriculum;
        #   and maybe with iterations_per_env (so that each horizon has same length still)

    def _evaluate(self, x, out, *args, **kwargs):
        print("Curric X =", x)
        curricula = self.evolCurric.evolXToCurriculum(x)
        print(curricula)
        # rewards = self.evolCurric.trainEveryCurriculum(curricula)
        rewards = []
        for i in range(len(curricula)):
            reward = 0
            for env in curricula[i]:
                if env == ENV_NAMES.DOORKEY_16x16:
                    reward += 10
            rewards.append(reward)
        out["F"] = -1 * np.array(rewards)
        print("EVALUATE PYMOO DONE")
