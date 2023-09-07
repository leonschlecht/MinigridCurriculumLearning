# RHEA CL (MiniGrid)

This repository contains the source code and materials for my master's thesis, which explores the combination of Curriculum Learning (CL) with Rolling Horizon Evolutionary Algorithms (RHEA). The PPO implementation and a rough overview of this repository are based on [lcswillems/rl-starter-files](https://github.com/lcswillems/rl-starter-files).

## Getting Started

### Prerequisites

Make sure you have the required dependencies installed. We used python 3.9
```bash
pip3 install -r requirements.txt
```

## Training an Agent
To train an agent using RHEA CL, run the following command:
```bash
python3 trainCurriculum.py --procs 24 --stepsPerCurric 3 --nGen 4 --numCurric 2 --iterPerEnv 100000 --model 150k_3step_4gen_2curric --noRewardShaping --seed 1
```

You can see a full list of parameters by using the --help flag. If you want to change the environments, update the ENV_NAMES.py file and the trainCurriculum.py file to select the desired environment.

## Evaluation Script
To evaluate the agent's performance and generate performance plots, use the following command as an example:
```bash
python3 evaluateCurriculum.py --iterGroup --rhea --scores curric
```

You can specify various options like --scores, --trainingTime, --curricDistr, or --splitDistr for filtering and comparisons. Filter options such as --rhea, --ga, --nsga, and --env are available. Note that some quick fixes have been implemented to make specific plots work.

## Visualization
After you trained an agent, you can visualize the agent's performance using the following command as an example:
```bash
python3 visualize.py --model <modelName/epoch_Nr> --env MiniGrid-DoorKey-8x8-v0
```


## Additional Notes
- If you cancel an experiment and wish to continue, it will start at the beginning of the last epoch, possibly requiring redoing 1 or 2 generations. Ensure that the arguments are correctly set.
- All logs and model files are stored in the /storage directory.
- You can train RHEA CL, Random RH, AllParallel (training all environments simultaneously), and SPCL.
- To view training rewards over time, use TensorBoard with the following command:
```bash
tensorboard --logdir path/To/Repo/storage
```
Sobol sampling points and max step reduction plots can be found in the ./helperPlots directory.


Feel free to explore the repository and experiment with the code for your research purposes.
