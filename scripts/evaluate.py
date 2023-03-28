import argparse
import time
import torch
from torch_ac.utils.penv import ParallelEnv
import json

import utils
from utils import device

# Parse arguments

parser = argparse.ArgumentParser()
parser.add_argument("--model", required=True,
                    help="name of the trained model (REQUIRED)")
parser.add_argument("--episodes", type=int, default=10,
                    help="number of episodes of evaluation (default: 10)")
parser.add_argument("--seed", type=int, default=0,
                    help="random seed (default: 0)")
parser.add_argument("--procs", type=int, default=16,
                    help="number of processes (default: 16)")
parser.add_argument("--argmax", action="store_true", default=False,
                    help="action with highest probability is selected")
parser.add_argument("--worst-episodes-to-show", type=int, default=10,
                    help="how many worst episodes to show")
parser.add_argument("--memory", action="store_true", default=False,
                    help="add a LSTM to the model")
parser.add_argument("--text", action="store_true", default=False,
                    help="add a GRU to the model")


def evaluateAgentInAllEnvs(evalEnv):
    # Load environments
    envs = []
    for i in range(args.procs):
        env = utils.make_env(evalEnv, args.seed + 10000 * i)
        envs.append(env)
    env = ParallelEnv(envs)
    # print("Environments loaded")

    # Load agent
    model_dir = utils.get_model_dir(args.model)
    agent = utils.Agent(env.observation_space, env.action_space, model_dir,
                        argmax=args.argmax, num_envs=args.procs,
                        use_memory=args.memory, use_text=args.text)
    # print("Agent loaded")

    # Initialize logs
    logs = {"num_frames_per_episode": [], "return_per_episode": []}

    # Run agent
    start_time = time.time()
    obss = env.reset()

    log_done_counter = 0
    log_episode_return = torch.zeros(args.procs, device=device)
    log_episode_num_frames = torch.zeros(args.procs, device=device)

    while log_done_counter < args.episodes:
        actions = agent.get_actions(obss)
        obss, rewards, terminateds, truncateds, _ = env.step(actions)
        dones = tuple(a | b for a, b in zip(terminateds, truncateds))
        agent.analyze_feedbacks(rewards, dones)

        log_episode_return += torch.tensor(rewards, device=device, dtype=torch.float)
        log_episode_num_frames += torch.ones(args.procs, device=device)

        for i, done in enumerate(dones):
            if done:
                log_done_counter += 1
                logs["return_per_episode"].append(log_episode_return[i].item())
                logs["num_frames_per_episode"].append(log_episode_num_frames[i].item())

        mask = 1 - torch.tensor(dones, device=device, dtype=torch.float)
        log_episode_return *= mask
        log_episode_num_frames *= mask

    end_time = time.time()

    # Print logs
    num_frames = sum(logs["num_frames_per_episode"])
    fps = num_frames / (end_time - start_time)
    duration = int(end_time - start_time)
    return_per_episode = utils.synthesize(logs["return_per_episode"])
    num_frames_per_episode = utils.synthesize(logs["num_frames_per_episode"])

    print("{} {} F {} | FPS {:.0f} | D {} | R:μσmM {:.2f} {:.2f} {:.2f} {:.2f} | F:μσmM {:.1f} {:.1f} {} {}"
          .format(evalEnv, args.model, num_frames, fps, duration,
                  *return_per_episode.values(),
                  *num_frames_per_episode.values()))

    mean = return_per_episode["mean"]
    maxRet = return_per_episode["max"]
    minRet = return_per_episode["min"]
    evaluationResult = {
        "meanRet": mean,
        "maxRet": maxRet,
        "minRet": minRet
    }

    env.reset()
    return evaluationResult


DOORKEY_5x5 = "MiniGrid-DoorKey-5x5-v0"
DOORKEY_6x6 = "MiniGrid-DoorKey-6x6-v0"
DOORKEY_8x8 = "MiniGrid-DoorKey-8x8-v0"
DOORKEY_16x16 = "MiniGrid-DoorKey-16x16-v0"
allEnvs = [DOORKEY_5x5, DOORKEY_6x6, DOORKEY_8x8, DOORKEY_16x16]


if __name__ == "__main__":
    args = parser.parse_args()
    utils.seed(args.seed)

    # print(f"Device: {device}")
    results = {"model": args.model}
    for evaluationEnv in allEnvs:
        results[evaluationEnv] = evaluateAgentInAllEnvs(evaluationEnv)
    json_object = json.dumps(results, indent=4)
    with open('storage/' + args.model + '/' + 'evaluation.json', 'w') as f:
        f.write(json_object)
    print(f"Evaluation of {args.model} succeeded")
