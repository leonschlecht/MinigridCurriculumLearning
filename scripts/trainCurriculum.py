from pymoo.core.result import Result

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
    objectives = 1
    curric1 = [ENV_NAMES.DOORKEY_6x6, ENV_NAMES.DOORKEY_5x5, ENV_NAMES.DOORKEY_5x5]
    curric2 = [ENV_NAMES.DOORKEY_8x8, ENV_NAMES.DOORKEY_16x16, ENV_NAMES.DOORKEY_16x16]
    xupper = len(ENV_NAMES.ALL_ENVS)
    curriculaList = [curric1]
    inequalityConstr = 0
    curricProblem = CurriculumProblem(curriculaList, objectives, inequalityConstr, xupper)

    algorithm = NSGA2(pop_size=len(curriculaList),
                      # sampling=BinaryRandomSampling(),
                      # crossover=TwoPointCrossover(),
                      # mutation=BitflipMutation(),
                      eliminate_duplicates=True
                      )

    res: Result = minimize(curricProblem,
                           algorithm,
                           ('n_gen', 50),  # terminationCriterion
                           seed=1,
                           verbose=True)  # True = print output during training

    # Scatter().add(res.F).show()
    print(res.F)
    print(res.X)
    print(curricProblem.curricula)
    print("We would append ", ENV_NAMES.ALL_ENVS[round(res.X[0])])

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
    def __init__(self, curricula: list, n_obj, n_ieq_constr, xu):
        n_var = len(curricula)
        # also wenn wir die Curricula (= je 1 individuum?), versuhcen wir die einzelnen listeneinträge element [xl, xu]
        # deshalb sollte das ein array sein; bzw. ist wahrscheinlich egal wenn es für alle gleich ist
        # Man könnte evtl mit ieq_constr ausprobieren, dass man homogene curricula vermeiden möchte oder so ?
        # Erstmal mit n_var = 1 ausprobieren; evtl später noch siherstellen dass n_var isch wirklich aus den curricula ergibt
        xl = np.zeros(n_var)
        xu = np.full(n_var, xu, dtype=int)
        print(xu)
        assert xl.shape == xu.shape
        super().__init__(n_var=n_var,  # the amount of curricula to use
                         n_obj=n_obj,  # maximizing the overall reward = 1 objective
                         n_ieq_constr=n_ieq_constr,  # 0 ?
                         xl=xl,
                         xu=xu)
        self.curricula = curricula
        self.N = 0
        # elementwise vs problem vs ?

        # das __evaluate könnte simulieren, was für rewards bestimmte Curricula bringen
        # Dwenn wir mehrere gens haben, wird das model währenddessen nicht (?) geupdated
        # alternativ könnte es sein, dass man den code in die epoch loo

        """
        epoch loop
            führe den RH aus
            führe Evolution aus, um zu bestimmten, wie die nächsten Curricula aussehen (input: aktueller stand des models,
                die rewards der aktuellen curricula; 
                die weiterführenden generationen würden dann den ganzen Trainingsprozess noch mal simulieren ???
                Falls nur 1 Generation: dann würde er versuchen zu bestimmen, was basierend auf: die curricula hatten die Rewards, mein Ziel ist es das zu maximieren. Wie sollte es nächstes mal aussehen?
            update das Model auf den snapshot fortschritt des besten Curriculums
        """


        """
        start evolution
        in der _evaluate:
            bestimmte den Reward der curricula (also ruf die train methode auf)
            update das model
            bestimmte die nächsten Curricula
            
            
        Was optimiert die evolution hier überhaupt ???
        """

        # F: what we want to maximize: ---> pymoo minimizes, so it should be -reward
        # G:# Inequality constraint;
        # H is EQ constraint: maybe we can experiment with the length of each curriculum;
        #   and maybe with iterations_per_env (so that each horizon has same length still)

    def _evaluate(self, x, out, *args, **kwargs):
        rewards = np.zeros(len(self.curricula))
        for i in range(len(self.curricula)):
            cur = self.curricula[i]
            rounded = round(x[i][0])
            # cur.append(ENV_NAMES.ALL_ENVS[rounded])
             # self.trainCurriculum(cur)
            rewards[i] = -200
        print(rewards)
        out["F"] = -rewards
        self.N += 1

    def trainCurriculum(self, cur):
        # this should be at the end of the evolutionary.py file in the train loop ? and it should be called ONCE per curricula I think
        print(1)
        return self.N


if __name__ == "__main__":
    registerEnvs()
    main()
    N = 0
