import argparse


def initializeArgParser():
    """
    Initializes the Argparser
    :return:
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--dynamicObstacle", action="store_true", default=False,
                        help="Whether to use dynamic obstacle or doorkey for training RHEA CL")
    parser.add_argument("--constMaxsteps", action="store_true", default=False,
                        help="Whether to keep maxSteps of environments constant or not")
    # AllParallel params
    parser.add_argument("--trainAllParalell", default=False, action="store_true",
                        help="traines all 4 environments in parallel")
    parser.add_argument("--asCurriculum", default=False, action="store_true",
                        help="Adds the option to use --allParalell as a linear curriculum (easy -> hard) per epoch")
    parser.add_argument("--ppoEnv", default=-1, type=int,
                        help="Adds the option to use --allParalell with PPO only for a single environment. Number corresponds to evironment index in array of all envs")
    parser.add_argument("--allSimultaneous", default=True, action="store_false",
                        help="If set, SPCL will be performed. --trainallParalell must be set too")
    parser.add_argument("--trainRandomRH", default=False, action="store_true",
                        help="Decides what training method will be used. If set, Full Random RH will be used")

    # RHEA CL Hyperparameters
    parser.add_argument("--iterPerEnv", default=150000, type=int,
                        help="Determines the amount of iterations per environment during training")
    parser.add_argument("--paraEnv", default=2, type=int,
                        help="The amount of envs to be trained on parallel at each timestep of the RH of a curriculum")
    parser.add_argument("--stepsPerCurric", default=3, type=int,
                        help="Determines the amount of steps used per curriculum during training. --paraEnv determines how many envs to be used")
    parser.add_argument("--numCurric", default=3, type=int,
                        help="Determines the amount of curricula that are used for training")
    parser.add_argument("--difficultyStepsize", default=100000, type=int, # TODO remove this ?
                        help="Determines when the difficulty will be adjusted. Default 100k -> -.1 decrease every 100k")  # TODO add starting @500k param too
    parser.add_argument("--trainingIterations", default=5000000, type=int,
                        help="How many RHEA CL Training iterations")
    parser.add_argument("--nGen", default=3, type=int,
                        help="The amount of generations per RHEA iteration")
    parser.add_argument("--gamma", default=0.9, type=float,
                        help="The dampening factor for RHEACL. Later steps will be weighed less if gamma is high")
    parser.add_argument("--noRewardShaping", action="store_true", default=False,
                        help="Whether or not to use rewardshaping for RHEA CL")
    # multiObj => use NSGA
    # EA PAram
    parser.add_argument("--useNSGA", default=False, action="store_true",
                        help="Decides what training method will be used. If set, adaptive curriculum will be used")
    parser.add_argument("--multiObj", default=False, action="store_true",
                        help="Whether multi objective NSGA is to be used")
    parser.add_argument("--crossoverProb", type=float, default=0.8, help="Crossover Probability of the RHEA CL")
    parser.add_argument("--mutationProb", type=float, default=0.8, help="Mutation Probability of the RHEA CL")
    parser.add_argument("--crossoverEta", type=float, default=3.0, help="Crossover ETA of the RHEA CL")
    parser.add_argument("--mutationEta", type=float, default=3.0, help="Mutation ETA of the RHEA CL")
    # General params
    parser.add_argument("--model", default=None, required=True, help="name of the model (REQUIRED)")
    parser.add_argument("--seed", type=int, default=1, help="random seed (default: 1)")
    parser.add_argument("--procs", type=int, default=32, help="number of processes (default: 32)")

    # PPO Parameters
    parser.add_argument("--epochs", type=int, default=4, help="number of epochs for PPO (default: 4)")
    parser.add_argument("--batch-size", type=int, default=256, help="batch size for PPO (default: 256)")
    parser.add_argument("--frames-per-proc", type=int, default=None,
                        help="number of frames per process before update (default: 5 for A2C and 128 for PPO)")
    parser.add_argument("--discount", type=float, default=0.99, help="discount factor (default: 0.99)")
    parser.add_argument("--lr", type=float, default=0.001, help="learning rate (default: 0.001)")
    parser.add_argument("--gae-lambda", type=float, default=0.95,
                        help="lambda coefficient in GAE formula (default: 0.95, 1 means no gae)")
    parser.add_argument("--entropy-coef", type=float, default=0.01, help="entropy term coefficient (default: 0.01)")
    parser.add_argument("--value-loss-coef", type=float, default=0.5, help="value loss term coefficient (default: 0.5)")
    parser.add_argument("--max-grad-norm", type=float, default=0.5, help="maximum norm of gradient (default: 0.5)")
    parser.add_argument("--optim-eps", type=float, default=1e-8,
                        help="Adam and RMSprop optimizer epsilon (default: 1e-8)")
    parser.add_argument("--optim-alpha", type=float, default=0.99, help="RMSprop optimizer alpha (default: 0.99)")
    parser.add_argument("--clip-eps", type=float, default=0.2, help="clipping epsilon for PPO (default: 0.2)")
    parser.add_argument("--recurrence", type=int, default=1,
                        help="number of time-steps gradient is backpropagated (default: 1). If > 1, a LSTM is added to the model to have memory.")
    parser.add_argument("--text", action="store_true", default=False,
                        help="add a GRU to the model to handle text input")

    # Evaluation Arguments
    parser.add_argument("--episodes", type=int, default=15,
                        help="number of episodes of evaluation per environment (default: 15)")
    parser.add_argument("--argmax", action="store_true", default=False,
                        help="action with highest probability is selected")
    parser.add_argument("--worst-episodes-to-show", type=int, default=10, help="how many worst episodes to show")
    parser.add_argument("--memory", action="store_true", default=False, help="add a LSTM to the model")

    # training logs / updates
    parser.add_argument("--log-interval", type=int, default=1,
                        help="number of updates between two logs (default: 1)")
    parser.add_argument("--save-interval", type=int, default=2,
                        help="number of updates between two saves (default: 2, 0 means no saving)")

    args = parser.parse_args()
    args.mem = args.recurrence > 1
    args.trainEvolutionary = not (args.trainRandomRH or args.trainAllParalell)

    # TODO create some logic to ensure proper usage and not using some wrong args params combinations
    # TODO create object for type safety
    return args
