from stable_baselines3 import PPO
from EnvironmentGridworld import CustomGridWorldEnv

#Create your gym environment
env = CustomGridWorldEnv()

# Load the model (optional, for demonstration)
model = PPO.load("ppo_custom_env_model", env=env)

# Test the trained agent
obs, _ = env.reset()
for _ in range(1000):
    action, _states = model.predict(obs, deterministic=True)
    obs, rewards, dones, truncated, info = env.step(action)
    print(f'pos: {env.agent_position}, action: {action}')
    env.render()