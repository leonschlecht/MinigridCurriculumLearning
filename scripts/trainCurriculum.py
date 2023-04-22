from datetime import datetime

import numpy as np
from gymnasium.envs.registration import register
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.core.evaluator import Evaluator
from pymoo.core.population import Population
from pymoo.core.problem import Problem

import utils
from curricula import linear, adaptive, EvolutionaryCurriculum
from utils import ENV_NAMES


def createFirstGeneration(curriculaList):
    indices = []
    for i in range(len(curriculaList)):
        indices.append([])
        for env in curriculaList[i]:
            indices[i].append(ENV_NAMES.ALL_ENVS.index(env))
    return indices


def tryEvolStuff():
    objectives = 1
    curric1 = [ENV_NAMES.DOORKEY_5x5, ENV_NAMES.DOORKEY_5x5, ENV_NAMES.DOORKEY_6x6, ENV_NAMES.DOORKEY_6x6]
    curric2 = [ENV_NAMES.DOORKEY_8x8, ENV_NAMES.DOORKEY_16x16, ENV_NAMES.DOORKEY_16x16, ENV_NAMES.DOORKEY_16x16]
    xupper = len(ENV_NAMES.ALL_ENVS)
    curriculaList = [curric2, curric1]
    inequalityConstr = 0
    curricProblem = CurriculumProblem(curriculaList, objectives, inequalityConstr, xupper)

    X = createFirstGeneration(curriculaList)
    pop = Population.new("X", X)
    Evaluator().eval(curricProblem, pop)

    algorithm = NSGA2(pop_size=len(curriculaList),
                      # sampling=BinaryRandomSampling(),
                      # crossover=TwoPointCrossover(),
                      # mutation=BitflipMutation(),
                      eliminate_duplicates=True,
                      )
    """
    res: Result = minimize(curricProblem,
                           algorithm,
                           ('n_gen', 300),  # terminationCriterion
                           seed=1,
                           verbose=False,
                           x0=curriculaList
                           )  # True = print output during training
    # Scatter().add(res.F).show()
    """

    # prepare the algorithm to solve the specific problem (same arguments as for the minimize function)
    algorithm.setup(curricProblem, termination=('n_gen', 300), seed=1, verbose=False)
    initialPop = Population.new("X", X)
    # Evaluator().eval(curricProblem, pop)
    # algorithm.evaluator.eval(curricProblem, initialPop)
    # algorithm.tell(infills=initialPop)

    while algorithm.has_next():
        # ask the algorithm for the next solution to be evaluated
        if initialPop is not None:
            pop = algorithm.ask()
        else:
            pop = initialPop
            initialPop = None
        # evaluate the individuals using the algorithm's evaluator (necessary to count evaluations for termination)
        algorithm.evaluator.eval(curricProblem, pop)
        # returned the evaluated individuals which have been evaluated or even modified

        algorithm.tell(infills=pop)

        # do same more things, printing, logging, storing or even modifying the algorithm object
        # print("n_gen, n_eval:", algorithm.n_gen, algorithm.evaluator.n_eval)
        print("----", "pop X:", np.round(pop.get("X")), pop.get("F"), "-----", sep="\n")
        # TODO update the model with best individual
    res = algorithm.result()
    print("hash", res.F.sum())
    print(np.round(res.X))

    return 0


# Select the best curriculum to show next
# best_curriculum_index = np.argmax(res.X)
# best_curriculum = curric[best_curriculum_index]


def main():
    args = utils.initializeArgParser()
    # TODO load time or set initially in linear / adaptive
    # TODO refactor to some utils method (for all methods)

    txtLogger = utils.get_txt_logger(utils.get_model_dir(args.model))

    # TODO fix iterations (so it doesnt overshoot the amount; maybe calculate with exact frame nrs or use updates)
    startTime = datetime.now()

    ############

    if args.trainEvolutionary:
        e = EvolutionaryCurriculum(txtLogger, startTime, args)
        tryEvolStuff()
    elif args.trainAdaptive:
        adaptive.adaptiveCurriculum(txtLogger, startTime, args)

    elif args.trainLinear:
        linear.startLinearCurriculum(txtLogger, startTime, args)


def registerEnvs():
    # TODO find better way so that the max_steps decrease over time
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
        assert len(curricula) > 0
        n_var = len(curricula[0])
        # maybe try to avoid homogenous curricula with ieq constraints (?)
        xl = np.full(n_var, -0.49)
        xu = np.full(n_var, xu - 0.51, dtype=float)
        super().__init__(n_var=n_var,  # the amount of curricula to use
                         n_obj=n_obj,  # maximizing the overall reward = 1 objective
                         n_ieq_constr=n_ieq_constr,  # 0 ?
                         xl=xl,
                         xu=xu)
        self.curricula = curricula
        self.N = 0
        # elementwise vs problem vs ?

        """
        (0) epoch loop
        (1)     führe den RH aus wie ich es bisher implementiert habe
        (2)     führe Evolution aus, um zu bestimmten, wie die nächsten Curricula aussehen (input: aktueller stand des models,
                    die rewards der aktuellen curricula, ... ? 
                    Was würde die _evaluate von pymoo denn aufrufen hier? Und was ist wenn Generationen > 1
        (3) update das Model auf den snapshot fortschritt des besten Curriculums, wie in (1) bestimmt
        """

        """
        start evolution durch minimize(...)
        in der _evaluate:
            bestimmte den Reward pro curriculum (also ruf die train methode für jedes curriculum auf)
            update das model
            bestimmte die nächsten Curricula            
        """
        # F: what we want to maximize: ---> pymoo minimizes, so it should be -reward
        # G:# Inequality constraint;
        # H is EQ constraint: maybe we can experiment with the length of each curriculum;
        #   and maybe with iterations_per_env (so that each horizon has same length still)

    def _evaluate(self, x, out, *args, **kwargs):
        rewards = np.zeros(len(self.curricula))
        # x should be of size len(curric) x len(curric[i]); so if we have 5 envs x 4 curric, it should be 5x4.
        # right now it is len(curric) x len(curric); every row represents one solution for the problem
        curricula = self.evolXToCurriculum(x)

        # TODO call the actual train method
        """
        for i in range(len(curricula)):
            cur = curricula[i]
            reward = 0
            for env in cur:
                if env == ENV_NAMES.DOORKEY_16x16:
                    reward += 10
            rewards[i] = reward
        """
        out["F"] = -rewards
        self.N += 1

    @staticmethod
    def evolXToCurriculum(x):
        result = []
        for i in range(x.shape[0]):
            result.append([])
            curric = x[i]
            for j in range(curric.shape[0]):
                rounded = round(x[i][j])
                result[i].append(ENV_NAMES.ALL_ENVS[rounded])
        return result


if __name__ == "__main__":
    registerEnvs()
    main()
