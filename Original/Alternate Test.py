import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt

from Brain import Brain, ReplayBuffer
import torch.optim as optim
import torch


def plot(name, x_lab, y_lab, data):
    plt.figure(figsize=(10, 6))
    plt.plot(data)
    plt.xlabel(x_lab)
    plt.ylabel(y_lab)
    #plt.savefig("average_reward_vs_episode.png")
    plt.title(name)
    plt.show()



num_episodes = 5000  # Number of episodes to train on
max_steps_per_episode = 200  # Max steps in an episode
learning_rate = 3e-4

env = gym.make('Pendulum-v1') # state = 3, action = 1
#env = gym.make('BipedalWalker-v3') # state = 24, action = 4
#env = gym.make('CartPole-v1')#, render_mode="human") # state = 4, action = 1 (discrete)


agent = Brain(learning_method = "SAC", optimizer_cls=optim.Adam, lr=learning_rate, batch_size=32, 
                 network_params = {"actor_state_size":3, "env_state_size":0, "output_size":1, "spatial":False,
                                   "nonspatial_depth":2, "nonspatial_width":8, "output_type":"distribution"})

#agent.load_model("./Models", "pendulum", learning_method="SAC")

memory = ReplayBuffer(32000)

episode_rewards_list = []
actor_losses = []
critic_losses = []

for episode in range(num_episodes):
    state, _ = env.reset()
    episode_rewards = 0

    if len(memory) > 1024 and episode % 50 == 0:
        losses = agent.learn(memory, 50)
        actor_loss = [loss[0] for loss in losses]
        critic_loss = [loss[1] for loss in losses]
        actor_losses.append(actor_loss)
        critic_losses.append(critic_loss)

    for step in range(max_steps_per_episode):
        #print(f'State: {state}')
        # Uncomment below if you want to see the training
        #env.render()

        with torch.no_grad():
            #action = np.array(agent.activate(torch.tensor(state), deterministic=False).ge(0.5).int())[0][0]
            action = np.array(((agent.activate(torch.tensor(state), deterministic=False) * 4) - 2)) #between -2 and 2 for pendulum
        #print(f'Action: {action}')
        next_state, reward, terminated, truncated, info = env.step(action)
        #print(f'Next State: {next_state}, Reward: {reward}, terminated: {terminated}')
        #print(f'terminated: {terminated}, Truncated: {truncated}')

        # Optionally, modify the reward based on the environment's feedback
        reward = (reward + 8) / 8


        memory.push(torch.reshape(torch.tensor(state), (3,-1)), torch.tensor(action), torch.tensor(reward),
                     torch.tensor(next_state), torch.tensor([terminated]))
        
        #print(f'STATE: {state.shape}')

        state = next_state
        episode_rewards += reward

        if terminated or truncated:
            break

        

    # Print episode's score
    episode_rewards_list.append(episode_rewards)
    #print(episode_rewards)
    if episode % 25 == 0:
        print(f"Episode: {episode}, Score: {sum(episode_rewards_list[-25:])/25}")
    

agent.save_model("./Models", "pendulum")

plot("Average Reward", "Episode", "Reward", episode_rewards_list)
plot("Actor Losses", "Episode", "Loss", actor_losses)
plot("Critic Losses", "Episode", "Loss", critic_losses)
