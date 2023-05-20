
"""
class Individual:
    # 	id, horizonLength, snapShotAfter, environment, <otherHyperparameters>
    def __init__(self, name, horizonLength, snapshotAfter, environment):
        self.name = name
        self.horizonLength = horizonLength
        self.snapshotAfter = snapshotAfter
        self.environment = environment

    def simulate(self, framesSoFar):
        frames = self.horizonLength + framesSoFar
        command = "python -m scripts.train --algo ppo --env {} --model {} --save-interval 10 --frames {}".\
            format(self.environment, "???", frames)
        print(command)


HORIZON_LENGTH = 500000
SNAPSHOT_AFTER = 100000

ind = Individual(0, HORIZON_LENGTH, SNAPSHOT_AFTER, environment="")
# ind.simulate(20000)

"""


"""

    lastReturn = -1

    SWITCH_AFTER = 50000
    switchAfter = SWITCH_AFTER
    UPDATES_BEFORE_SWITCH = 5
    updatesLeft = UPDATES_BEFORE_SWITCH
"""

"""
            currentReturn = rreturn_per_episode["mean"]
            value = logs["value"]
            converged = value >= 0.78
            policyLoss = logs["policy_loss"]
            performanceDecline = lastReturn >= currentReturn + .1 or currentReturn < 0.1 or \
                                 value < 0.05 or abs(policyLoss) < 0.02

            if abs(policyLoss) < 0.03:
                currentEnv = DOORKEY_8x8
                algo = setAlgo(envs8x8, acmodel, preprocess_obss8x8)
                switchAfter = SWITCH_AFTER
                framesWithThisEnv = 0
            if converged:
                updatesLeft -= 1
            if updatesLeft < UPDATES_BEFORE_SWITCH:
                updatesLeft -= 1
                if updatesLeft < 0:
                    currentEnv = nextEnv(currentEnv)
                    algo = setAlgo(envs16x16, acmodel, preprocess_obss)
                    updatesLeft = UPDATES_BEFORE_SWITCH
                    switchAfter = SWITCH_AFTER
                    framesWithThisEnv = 0
            lastReturn = currentReturn

"""
