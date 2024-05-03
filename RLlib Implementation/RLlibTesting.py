import ray
from ray.rllib.algorithms.ppo import PPOConfig
from ray import tune
from EnvironmentRLlib import Environment

# Initialize Ray
ray.init()

tune.register_env("Maze-v0", Environment)

agent = (
    PPOConfig()
    .rollouts(num_rollout_workers=1)
    .resources(num_gpus=1)
    .environment(env='Maze-v0')
    .build()
)


# Path to the saved checkpoint (replace with your checkpoint path)
checkpoint_path = "./Checkpoints/FirstModel"

# Load the trained model from checkpoint
agent.restore(checkpoint_path)

# Assuming you have an environment instance
env = Environment()

# Run the model on the environment
done = False
keep_going = "y"
while keep_going == "y":
    obs, _ = env.reset()
    while not done:
        action = agent.compute_single_action(obs)
        obs, reward, done, truncated, info = env.step(action)
        env.render()
    keep_going = input("do another run? [y,n]")
