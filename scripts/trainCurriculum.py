import utils
from utils import ENV_NAMES
from curricula import linear, adaptive, EvolutionaryCurriculum
from datetime import datetime
from gymnasium.envs.registration import register
import numpy as np
from pymoo.core.problem import Problem
from pymoo.operators.crossover.pntx import TwoPointCrossover
from pymoo.operators.mutation.bitflip import BitflipMutation
from pymoo.operators.sampling.rnd import BinaryRandomSampling
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.problems import get_problem
from pymoo.optimize import minimize
from pymoo.visualization.scatter import Scatter

from utils import ENV_NAMES


def tryEvolStuff():
    return 0

# Select the best curriculum to show next
# best_curriculum_index = np.argmax(res.X)
# best_curriculum = curric[best_curriculum_index]


def main():
    args = utils.initializeArgParser()
    # TODO load time or set initially in linear / adaptive
    # TODO refactor to some utils method (for all methods)

    txtLogger = utils.get_txt_logger(utils.get_model_dir(args.model))

    # TODO limit max frames per env in evaluation
    # TODO fix iterations (so it doesnt overshoot the amount; maybe calculate with exact frame nrs or use updates)
    startTime = datetime.now()

    ############

    tryEvolStuff()
    exit(999)
    if args.trainEvolutionary:
        e = EvolutionaryCurriculum(txtLogger, startTime, args)
    elif args.trainAdaptive:
        adaptive.adaptiveCurriculum(txtLogger, startTime, args)

    elif args.trainLinear:
        linear.startLinearCurriculum(txtLogger, startTime, args)


def registerEnvs():
    register(
        id="MiniGrid-Empty-Random-8x8-v0",  # todo use from ENV_NAMES
        entry_point="minigrid.envs:EmptyEnv",
        kwargs={"size": 8, "agent_start_pos": None},
    )

    register(
        id="MiniGrid-Empty-Random-16x16-v0",
        entry_point="minigrid.envs:EmptyEnv",
        kwargs={"size": 16, "agent_start_pos": None},
    )

    register(
        id="MiniGrid-DoorKey-16x16-custom",
        entry_point="minigrid.envs:DoorKeyEnv",
        kwargs={"size": 16, "max_steps": 300},
    )

    register(
        id="MiniGrid-DoorKey-8x8-custom",
        entry_point="minigrid.envs:DoorKeyEnv",
        kwargs={"size": 8, "max_steps": 200},
    )

    register(
        id="MiniGrid-DoorKey-6x6-custom",
        entry_point="minigrid.envs:DoorKeyEnv",
        kwargs={"size": 6, "max_steps": 125},
    )

    register(
        id="MiniGrid-DoorKey-5x5-custom",
        entry_point="minigrid.envs:DoorKeyEnv",
        kwargs={"size": 5, "max_steps": 100},
    )


class CurriculumProblem(Problem):
    def __init__(self, n_var, n_obj, n_ieq_constr, xl, xu, curricula: list):
        #        super().__init__(n_var=10, n_obj=1, n_ieq_constr=1, xl=0.0, xu=1.0)
        super().__init__(n_var=n_var,
                         n_obj=n_obj,
                         n_ieq_constr=n_ieq_constr,
                         xl=xl,
                         xu=xu)
        self.curricula = curricula
        self.N = 0
        # n_var = curriculum length
        # n_ieq_ ... should be same as n_var ?
        # n_obj = 1 (maximize all rewards)
        # from getting started: elementwise = 1/3 ways of doing this

        # F: what we want to maximize: ---> pymoo minimizes, so it should be -reward
        # G:# IEQ constraint; H is EQ constraint

    def _evaluate(self, x, out, *args, **kwargs):
        print(x.shape)
        rewards = np.zeros(len(self.curricula))
        for i in range(len(self.curricula)):
            cur = self.curricula[i]
            cur.append(x[i])
            reward = self.trainCurriculum(cur)
            rewards[i] = reward
        out["F"] = -rewards
        print(out)
        self.N += 1
        if self.N > 10:
            exit(self.N)

    def trainCurriculum(self, cur):
        # Implement your training function here
        print(1)
        return self.N  # this should be at the end of the evolutionary.py file in the train loop ? and it should be called ONCE per curricula I think


if __name__ == "__main__":
    registerEnvs()
    main()
    N = 0
