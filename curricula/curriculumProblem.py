from pymoo.core.evaluator import Evaluator
from pymoo.core.problem import Problem
import numpy as np

from curricula import RollingHorizonEvolutionaryAlgorithm
from utils import ENV_NAMES


class CurriculumProblem(Problem):
    def __init__(self, curricula: list, n_obj, n_ieq_constr, xu, paraEnvs: int,
                 rheaObj: RollingHorizonEvolutionaryAlgorithm):
        assert len(curricula) > 0
        assert rheaObj is not None
        self.rheaObj = rheaObj
        n_var = len(curricula[0]) * paraEnvs
        xl = np.zeros(n_var, dtype=int)
        xu = np.full(n_var, xu, dtype=int)
        super().__init__(n_var=n_var,
                         n_obj=n_obj,  # maximizing the overall reward = 1 objective
                         n_ieq_constr=n_ieq_constr,  # = 0 ?
                         xl=xl,
                         xu=xu)
        self.curricula = curricula
        self.gen = 0
        # TODO maybe try to avoid homogenous curricula with ieq constraints (?)

        # F: what we want to maximize: ---> pymoo minimizes, so it should be -reward
        # G:# Inequality constraint;
        # H is EQ constraint: maybe we can experiment with the length of each curriculum;
        #   and maybe with iterations_per_env (so that each horizon has same length still)

    def _evaluate(self, x, out, *args, **kwargs):
        self.gen += 1
        rewards = self.rheaObj.trainEveryCurriculum(x, self.gen)
        # rewards = self.dummyRewards(curricula)
        out["F"] = -1 * np.array(rewards)

    def dummyRewards(self, curricula):
        rewards = []
        for i in range(len(curricula)):
            reward = 0
            for env in curricula[i]:
                if env == ENV_NAMES.DOORKEY_16x16:
                    reward += 10
            rewards.append(reward)
        self.rheaObj.currentRewardsDict = rewards
        return rewards
