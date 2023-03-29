import argparse
import time
import torch
from torch_ac.utils.penv import ParallelEnv
import json

import utils
from utils import device
from constants import ENV_NAMES


def evaluateAgentInAllEnvs(args, model, evalEnv) -> dict:
    # Load environments
    envs = []
    for i in range(args.procs):
        env = utils.make_env(evalEnv, args.seed + 10000 * i)
        envs.append(env)
    env = ParallelEnv(envs)

    # Load agent
    model_dir = utils.get_model_dir(model)
    agent = utils.Agent(env.observation_space, env.action_space, model_dir,
                        argmax=args.argmax, num_envs=args.procs,
                        use_memory=args.memory, use_text=args.text)

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

    print(
        "EVAL: {} with {} : F {} | FPS {:.0f} | D {} | R:μσmM {:.2f} {:.2f} {:.2f} {:.2f} | F:μσmM {:.1f} {:.1f} {} {}"
        .format(evalEnv, model, num_frames, fps, duration,
                *return_per_episode.values(),
                *num_frames_per_episode.values()))

    evaluationResult = {
        "meanRet": return_per_episode["mean"],
        "maxRet": return_per_episode["max"],
        "minRet": return_per_episode["min"]
    }

    env.reset()
    return evaluationResult


def evaluateAll(model, args) -> dict:
    utils.seed(args.seed)
    results = {"model": model}
    for evaluationEnv in ENV_NAMES.ALL_ENVS:
        results[evaluationEnv] = evaluateAgentInAllEnvs(args, model, evaluationEnv)
    evaluationResults: str = json.dumps(results, indent=4)
    with open('storage/' + model + '/' + 'evaluation.json', 'w') as f:
        f.write(evaluationResults)
    print(f"Evaluation of {model} succeeded")
    return results
