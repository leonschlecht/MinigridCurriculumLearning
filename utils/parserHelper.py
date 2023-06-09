import argparse


def initializeArgParser():
    """
    Initializes the Argparser
    :return:
    """
    parser = argparse.ArgumentParser()
    # General parameters
    parser.add_argument("--trainAdaptive", default=False, action="store_true",
                        help="Decides what training method will be used. If set, adaptive curriculum will be used")
    parser.add_argument("--trainAllParalell", default=False, action="store_true",
                        help="traines all 4 environments in parallel")
    parser.add_argument("--allSimultaneous", default=True, action="store_false",
                        help="Determines if all envs should be trained simultaneously from the start, or if the envs should be selected depending on progress")
    parser.add_argument("--trainLinear", default=False, action="store_true",
                        help="Decides what training method will be used. If set, linear curriculum will be used")
    parser.add_argument("--trainBiasedRandomRH", default=False, action="store_true",
                        help="Decides what training method will be used. If set, Biased Random RH will be used")
    parser.add_argument("--trainRandomRH", default=False, action="store_true",
                        help="Decides what training method will be used. If set, Full Random RH will be used")

    parser.add_argument("--iterPerEnv", default=150000, type=int,
                        help="Determines the amount of iterations per environment during training")
    parser.add_argument("--paraEnv", default=2, type=int,
                        help="The amount of envs to be trained on parallel at each timestep of the RH of a curriculum")
    parser.add_argument("--stepsPerCurric", default=3, type=int,
                        help="Determines the amount of steps used per curriculum during training. --paraEnv determines how many envs to be used")
    parser.add_argument("--numCurric", default=3, type=int,
                        help="Determines the amount of curricula that are used for training")
    parser.add_argument("--difficultyStepsize", default=100000, type=int,
                        help="Determines when the difficulty will be adjusted. Default 100k -> -.1 decrease every 100k")
    parser.add_argument("--trainEpochs", default=25, type=int, help="Tells the algorithm how long to train for.")
    parser.add_argument("--nGen", default=3, type=int,
                        help="The amount of generations per RHEA iteration")
    parser.add_argument("--gamma", default=0.9, type=float,
                        help="The dampening factor for the curricula RH. Later steps will be weighed less if gamma is high")

    parser.add_argument("--algo", default="ppo", help="algorithm to use: a2c | ppo ")
    parser.add_argument("--model", default=None, required=True, help="name of the model (REQUIRED)")
    parser.add_argument("--seed", type=int, default=1, help="random seed (default: 1)")
    parser.add_argument("--log-interval", type=int, default=1, help="number of updates between two logs (default: 1)")
    parser.add_argument("--save-interval", type=int, default=2,
                        help="number of updates between two saves (default: 2, 0 means no saving)")
    parser.add_argument("--procs", type=int, default=32, help="number of processes (default: 32)")

    # Parameters for underlying training algorithm
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
    parser.add_argument("--episodes", type=int, default=25, help="number of episodes of evaluation (default: 15)")
    parser.add_argument("--argmax", action="store_true", default=False,
                        help="action with highest probability is selected")
    parser.add_argument("--worst-episodes-to-show", type=int, default=10, help="how many worst episodes to show")
    parser.add_argument("--memory", action="store_true", default=False, help="add a LSTM to the model")

    args = parser.parse_args()
    args.mem = args.recurrence > 1
    args.trainEvolutionary = not (
            args.trainLinear or args.trainAdaptive or args.trainRandomRH or args.trainBiasedRandomRH or args.trainAllParalell
    )
    # TODO create object for type safety
    return args
