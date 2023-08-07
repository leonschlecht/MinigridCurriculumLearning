import argparse
import numpy
from gymnasium.envs.registration import register

import utils
from utils import device, ENV_NAMES

# Parse arguments

parser = argparse.ArgumentParser()
parser.add_argument("--model", required=True, help="name of the trained model (REQUIRED)")
parser.add_argument("--seed", type=int, default=0, help="random seed (default: 0)")
parser.add_argument("--shift", type=int, default=0,
                    help="number of times the environment is reset at the beginning (default: 0)")
parser.add_argument("--argmax", action="store_true", default=False,
                    help="select the action with highest probability (default: False)")
parser.add_argument("--pause", type=float, default=0.1,
                    help="pause duration between two consequent actions of the agent (default: 0.1)")
parser.add_argument("--gif", type=str, default=None, help="store output as gif with the given filename")
parser.add_argument("--episodes", type=int, default=1000000, help="number of episodes to visualize")
parser.add_argument("--memory", action="store_true", default=False, help="add a LSTM to the model")
parser.add_argument("--text", action="store_true", default=False, help="add a GRU to the model")
parser.add_argument("--size", type=int, help="Tell the level size of the environment")
parser.add_argument("--env", type=str, default=None, help="What env to visualize")

args = parser.parse_args()

# Set seed for all randomness sources

utils.seed(args.seed)

# Set device

print(f"Device: {device}\n")
if args.env is None:
    envName = "MiniGrid-DoorKey-custom-" + str(args.size) + "x" + str(args.size)
    envName = "MiniGrid-PutNear-8x8-N3-v0"
    envName = "MiniGrid-DistShift2-v0"
    envName = "MiniGrid-MultiRoom-N4-S5-v0"
else:
    envName = args.env

env = utils.make_env(envName, args.seed, render_mode="human")
for _ in range(args.shift):
    env.reset()
print("Environment loaded\n")

# Load agent

model_dir = utils.get_model_dir(args.model)
agent = utils.Agent(env.observation_space, env.action_space, model_dir,
                    argmax=args.argmax, use_memory=args.memory, use_text=args.text)
print("Agent loaded\n")

# Run the agent

if args.gif:
    from array2gif import write_gif

    frames = []

# Create a window to view the environment
env.render()

for episode in range(args.episodes):
    obs, _ = env.reset()

    while True:
        env.render()
        if args.gif:
            frames.append(numpy.moveaxis(env.get_frame(), 2, 0))

        action = agent.get_action(obs)
        obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated | truncated
        agent.analyze_feedback(reward, done)

        if done or env.window.closed:
            break

    if env.window.closed:
        break

if args.gif:
    print("Saving gif... ", end="")
    write_gif(numpy.array(frames), args.gif + ".gif", fps=1 / args.pause)
    print("Done.")
