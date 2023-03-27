

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
ind.simulate(20000)
