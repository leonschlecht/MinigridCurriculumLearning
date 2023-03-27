import os
import argparse
# import train


DOORKEY_5x5 = "MiniGrid-DoorKey-5x5"
DOORKEY_6x6 = "MiniGrid-DoorKey-6x6"
DOORKEY_8x8 = "MiniGrid-DoorKey-8x8"
DOORKEY_16x16 = "MiniGrid-DoorKey-16x16-v0"
allEnvs = [DOORKEY_5x5, DOORKEY_6x6, DOORKEY_8x8, DOORKEY_16x16]

parser = argparse.ArgumentParser()
# General parameters
parser.add_argument("--algo", default="ppo",
                    help="algorithm to use: a2c | ppo (REQUIRED)")
parser.add_argument("--env",
                    help="environemnt ") # TODO
parser.add_argument("--model", default=None,
                    help="name of the model (default: {ENV}_{ALGO}_{TIME})")
parser.add_argument("--seed", type=int, default=1,
                    help="random seed (default: 1)")
parser.add_argument("--log-interval", type=int, default=1,
                    help="number of updates between two logs (default: 1)")
parser.add_argument("--save-interval", type=int, default=10,
                    help="number of updates between two saves (default: 10, 0 means no saving)")
parser.add_argument("--procs", type=int, default=16,
                    help="number of processes (default: 16)")
parser.add_argument("--frames", type=int, default=10 ** 7,
                    help="number of frames of training (default: 1e7)")

# Parameters for main algorithm
parser.add_argument("--epochs", type=int, default=4,
                    help="number of epochs for PPO (default: 4)")
parser.add_argument("--batch-size", type=int, default=256,
                    help="batch size for PPO (default: 256)")
parser.add_argument("--frames-per-proc", type=int, default=None,
                    help="number of frames per process before update (default: 5 for A2C and 128 for PPO)")
parser.add_argument("--discount", type=float, default=0.99,
                    help="discount factor (default: 0.99)")
parser.add_argument("--lr", type=float, default=0.001,
                    help="learning rate (default: 0.001)")
parser.add_argument("--gae-lambda", type=float, default=0.95,
                    help="lambda coefficient in GAE formula (default: 0.95, 1 means no gae)")
parser.add_argument("--entropy-coef", type=float, default=0.01,
                    help="entropy term coefficient (default: 0.01)")
parser.add_argument("--value-loss-coef", type=float, default=0.5,
                    help="value loss term coefficient (default: 0.5)")
parser.add_argument("--max-grad-norm", type=float, default=0.5,
                    help="maximum norm of gradient (default: 0.5)")
parser.add_argument("--optim-eps", type=float, default=1e-8,
                    help="Adam and RMSprop optimizer epsilon (default: 1e-8)")
parser.add_argument("--optim-alpha", type=float, default=0.99,
                    help="RMSprop optimizer alpha (default: 0.99)")
parser.add_argument("--clip-eps", type=float, default=0.2,
                    help="clipping epsilon for PPO (default: 0.2)")
parser.add_argument("--recurrence", type=int, default=1,
                    help="number of time-steps gradient is backpropagated (default: 1). If > 1, a LSTM is added to the model to have memory.")
parser.add_argument("--text", action="store_true", default=False,
                    help="add a GRU to the model to handle text input")


# TODO copy back
import time
import torch_ac
import tensorboardX
import sys

import utils
from utils import device
from model import ACModel

def main(args):
    print(args)
    # Set run dir
    model_name = args.model
    model_dir = utils.get_model_dir(model_name)

    # Load loggers and Tensorboard writer
    txt_logger = utils.get_txt_logger(model_dir)
    csv_file, csv_logger = utils.get_csv_logger(model_dir)
    tb_writer = tensorboardX.SummaryWriter(model_dir)

    # Log command and all script arguments
    txt_logger.info("{}\n".format(" ".join(sys.argv)))
    txt_logger.info("{}\n".format(args))

    # Set seed for all randomness sources
    utils.seed(args.seed)

    # Set device
    txt_logger.info(f"Device: {device}\n")

    # Load environments
    envs = []
    for i in range(args.procs):
        envs.append(utils.make_env(args.env, args.seed + 10000 * i))
    txt_logger.info("Environments loaded\n")

    # Load training status
    try:
        status = utils.get_status(model_dir)
    except OSError:
        status = {"num_frames": 0, "update": 0}
    txt_logger.info("Training status loaded\n")

    # Load observations preprocessor
    obs_space, preprocess_obss = utils.get_obss_preprocessor(envs[0].observation_space)

    if "vocab" in status:
        preprocess_obss.vocab.load_vocab(status["vocab"])
    txt_logger.info("Observations preprocessor loaded")

    # Load model
    acmodel = ACModel(obs_space, envs[0].action_space, args.mem, args.text)

    if "model_state" in status:
        acmodel.load_state_dict(status["model_state"])
    acmodel.to(device)
    txt_logger.info("Model loaded\n")
    txt_logger.info("Acmodel 16x16: {}\n".format(acmodel))

    # Load algo
    start = time.time()
    print("Loading algorithm. . . ")
    algo = torch_ac.PPOAlgo(envs, acmodel, device, args.frames_per_proc, args.discount, args.lr, args.gae_lambda,
                            args.entropy_coef, args.value_loss_coef, args.max_grad_norm, args.recurrence,
                            args.optim_eps, args.clip_eps, args.epochs, args.batch_size, preprocess_obss)

    print("Algorithm loaded in", (round(-start + time.time(), 2), "sec"))

    if "optimizer_state" in status:
        algo.optimizer.load_state_dict(status["optimizer_state"])
    txt_logger.info("Optimizer loaded\n")

    # Train model
    num_frames = status["num_frames"]
    update = status["update"]
    start_time = time.time()

    framesWithThisEnv = 0

    while num_frames < args.frames:
        update_start_time = time.time()

        exps, logs1 = algo.collect_experiences()
        logs2 = algo.update_parameters(exps)
        logs = {**logs1, **logs2}
        update_end_time = time.time()

        framesWithThisEnv += logs["num_frames"]

        num_frames += logs["num_frames"]
        update += 1

        # Print logs
        if update % args.log_interval == 0:
            fps = logs["num_frames"] / (update_end_time - update_start_time)
            duration = int(time.time() - start_time)
            return_per_episode = utils.synthesize(logs["return_per_episode"])
            rreturn_per_episode = utils.synthesize(logs["reshaped_return_per_episode"])
            num_frames_per_episode = utils.synthesize(logs["num_frames_per_episode"])

            header = ["update", "frames", "FPS", "duration"]
            data = [update, num_frames, fps, duration]
            header += ["rreturn_" + key for key in rreturn_per_episode.keys()]
            data += rreturn_per_episode.values()
            header += ["num_frames_" + key for key in num_frames_per_episode.keys()]
            data += num_frames_per_episode.values()
            header += ["entropy", "value", "policy_loss", "value_loss", "grad_norm"]
            data += [logs["entropy"], logs["value"], logs["policy_loss"], logs["value_loss"], logs["grad_norm"]]

            txt_logger.info(
                "{} {} U {} | F {:06} | FPS {:04.0f} | D {} | rR:msmM {:.3f} {:.2f} {:.2f} {:.2f} | F:msmM {:.1f} {:.1f} {} {} | H {:.2f} | V {:.4f} | pL {:.4f} | vL {:.4f} | g {:.4f}"
                .format(args.env, framesWithThisEnv, *data))

            header += ["return_" + key for key in return_per_episode.keys()]
            data += return_per_episode.values()

            if status["num_frames"] == 0:
                csv_logger.writerow(header)
            csv_logger.writerow(data)
            csv_file.flush()

            for field, value in zip(header, data):
                tb_writer.add_scalar(field, value, num_frames)

        # Save status
        if update % args.save_interval == 0:
            status = {"num_frames": num_frames, "update": update,
                      "model_state": acmodel.state_dict(), "optimizer_state": algo.optimizer.state_dict()}
            if hasattr(preprocess_obss, "vocab"):
                status["vocab"] = preprocess_obss.vocab.vocab
            utils.save_status(status, model_dir)
            txt_logger.info("Status saved")

    # Save status  after training is done
    if update % args.save_interval == 0:
        status = {"num_frames": num_frames, "update": update,
                  "model_state": acmodel.state_dict(), "optimizer_state": algo.optimizer.state_dict()}
        if hasattr(preprocess_obss, "vocab"):
            status["vocab"] = preprocess_obss.vocab.vocab
        utils.save_status(status, model_dir)
        txt_logger.info("Status saved")



############
def evaluateAgent(model):
    reward = 0
    for testEnv in allEnvs:
        os.system("python -m scripts.evaluate --episodes 4 --procs 32 --env " + testEnv + " --model " + model)
        # TODO: Read Json contents, then test this
        with open('storage/' + model + '/' + testEnv + '_evaluation.json', 'r') as f:
            text = f.readlines()
            print(text)
            reward += text["meanRet"]
            # reward = tex["meanRet"] # ?

    return reward


def nextEnv(currentEnv):
    if currentEnv == DOORKEY_8x8:
        return DOORKEY_16x16
    if currentEnv == DOORKEY_6x6:
        return DOORKEY_8x8
    if currentEnv == DOORKEY_5x5:
        return DOORKEY_6x6
    raise Exception("Next Env ??")


def prevEnv(currentEnv):
    if currentEnv == DOORKEY_16x16:
        return DOORKEY_8x8
    if currentEnv == DOORKEY_8x8:
        return DOORKEY_6x6
    if currentEnv == DOORKEY_6x6:
        return DOORKEY_5x5
    return DOORKEY_5x5


def train(frames, env, model):
    args.env = env
    args.model = model
    args.frames = frames
    main(args)
    print('Trained ' + args.env + ' using ppo to folder ' + model + ' with ' + str(0) + ' frames')
    time.sleep(1)


def trainEnv():
    HORIZON_LENGTH = 400000
    ITERATIONS_PER_CURRICULUM = 100000
    frames = ITERATIONS_PER_CURRICULUM  # TODO load frame status and set it
    modelName = args.model

    for i in range(HORIZON_LENGTH // ITERATIONS_PER_CURRICULUM):
        break
        if i % 2 == 0:
            env = DOORKEY_5x5
        else:
            env = DOORKEY_6x6
        train(frames * (i+1), env, modelName)
    rewards = evaluateAgent(modelName)
    print("Rewards = ", rewards)


if __name__ == "__main__":
    args = parser.parse_args()
    args.mem = args.recurrence > 1
    trainEnv()

"""
    lastReturn = -1

    SWITCH_AFTER = 50000
    switchAfter = SWITCH_AFTER
    UPDATES_BEFORE_SWITCH = 5
    updatesLeft = UPDATES_BEFORE_SWITCH
    framesWithThisEnv = 0
    performanceDecline = False
    converged = False
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
