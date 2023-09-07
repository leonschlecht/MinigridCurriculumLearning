from pymoo.core.evaluator import Evaluator
from pymoo.core.problem import Problem
import numpy as np

from curricula import RollingHorizonEvolutionaryAlgorithm
from utils import ENV_NAMES


class CurriculumProblem(Problem):
    """
    The Problem used for the PYMOO optimization in RHEA CL
    """
    def __init__(self, curricula: list, n_obj, n_ieq_constr, xu, paraEnvs: int,
                 rheaObj: RollingHorizonEvolutionaryAlgorithm):
        """
        :param curricula: the list of curricula
        :param n_obj: the amount of objectives you want to optimize
        :param n_ieq_constr: the amount of inequality constraints
        :param xu: the upper limit for x. (The lower limit is assumed 0)
        :param paraEnvs: The amount of parallel envs to be trained on
        :param rheaObj: the reference to the RHEA class
        """
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

    def _evaluate(self, x, out, *args, **kwargs):
        self.gen += 1
        rewards = self.rheaObj.trainEveryCurriculum(x, self.gen)
        # rewards = self.dummyRewards(curricula)
        out["F"] = -1 * np.array(rewards)

    def dummyRewards(self, curricula):
        """
        Deprecated. Do not use anymore
        Old helper method that was used to test PYMOO and seeing if the dimensions for each parameter etc were set correctly
        :param curricula:
        :return:
        """
        rewards = []
        for i in range(len(curricula)):
            reward = 0
            for env in curricula[i]:
                if env == ENV_NAMES.DOORKEY_16x16:
                    reward += 10
            rewards.append(reward)
        self.rheaObj.currentRewardsDict = rewards
        return rewards
