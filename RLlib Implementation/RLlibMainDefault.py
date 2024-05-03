from ray import tune
from gym.envs.registration import register
from ray.rllib.models import ModelCatalog
import ray
from ray.rllib.algorithms import ppo
import matplotlib.pyplot as plt

from EnvironmentRLlib import Environment



'''Training on the maze environment using a default model'''

def plot(name, x_lab, y_lab, data):
    plt.figure(figsize=(10, 6))
    plt.plot(data)
    plt.xlabel(x_lab)
    plt.ylabel(y_lab)
    plt.title(name)
    plt.show()


ray.init()

tune.register_env("Maze-v0", Environment)

algo = ppo.PPO(env="Maze-v0", config = {})

# config = SACConfig().training(gamma=0.9, lr=0.01, train_batch_size=32)
# config = config.resources(num_gpus=1)
# config = config.rollouts(num_rollout_workers=2)

# # Build a Algorithm object from the config and run 1 training iteration.
# algo = config.build(env="Maze-v0")
        
custom_checkpoint_dir = "./Checkpoints/RandStart"
algo.restore(custom_checkpoint_dir)
custom_checkpoint_dir = "./Checkpoints/SetStart"

print_guide = ["episodes_this_iter", "episode_reward_mean", "episode_reward_max", "episode_reward_min",
               "done", "num_env_steps_sampled_this_iter", ]

mean_rewards = []

print("BEGINNING TRAINING")
for i in range(1, 501):
    result = algo.train()
    mean_rewards.append(result["episode_reward_mean"])

    print(f"Iteration: {i}")
    for key in print_guide:
        print(f'{key}: {result[key]}')

    if i % 20 == 0:
        checkpoint_dir = algo.save(custom_checkpoint_dir).checkpoint.path
        print(f"Checkpoint saved in directory {checkpoint_dir}")
        
    print("\n")

plot("Mean Rewards", "Mean Rewards", "Training Iteration", mean_rewards)
