from ray import tune
from gym.envs.registration import register
from ray.rllib.models import ModelCatalog
import ray
from ray.rllib.algorithms import PPOConfig
import matplotlib.pyplot as plt

# Assuming KoopmanResNet is defined elsewhere and is importable
from KResNetRLlib import KoopmanResNet
from EnvironmentRLlib import Environment

'''Training on the maze environment using the koopman residual neural network'''

def plot(name, x_lab, y_lab, data):
    plt.figure(figsize=(10, 6))
    plt.plot(data)
    plt.xlabel(x_lab)
    plt.ylabel(y_lab)
    plt.title(name)
    plt.show()

# Register your environment
register(
    id='MazeEnv-v0',  # Make sure the ID is correctly formatted
    entry_point='your_module:Environment',  # Correct the entry_point format
)

# Register your custom model
ModelCatalog.register_custom_model("KoopmanResNet", KoopmanResNet)

ray.init()

tune.register_env("MazeEnv-v0", Environment)

algo = (
        PPOConfig()
        .environment(env="MazeEnv-v0")
        .resources(num_gpus=0)
        .training(#lr=0.001, grad_clip=30.0, 
                  model={
                    "custom_model": "KoopmanResNet",
                    "custom_model_config": {
                        "nLayers": 2,
                        "actionSize": 3,
                        "encodingDims": 16,
                        "inChannels": 1,
                        "resChannels": 1,
                        "dropout": 0,
                        "batchNorm": False,
                        "activation": "relu",
                        "initialization": "xh",
                        "inputChannels2D": 7 * 7, 
                        "inputChannels1D": 7
                    },
                  }
    
        )
        .build()
        )
        
    

custom_checkpoint_dir = "./Checkpoints/May_2_Model"


print_guide = ["episodes_this_iter", "episode_reward_mean", "episode_reward_max", "episode_reward_min",
               "policy_reward_mean", "done", 
               "num_env_steps_sampled_this_iter", ]

mean_rewards = []

print("BEGINNING TRAINING")
for i in range(1, 21):
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
