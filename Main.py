from Environment import *
from Brain import *

import random
import matplotlib.pyplot as plt

class GraphTool():
    def __init__(self, name):
        self.name = name
        self.data = []

    def push(self, datum):
        self.data.append(datum)

    def plot(self, y_label, x_label="Episodes"):
        plt.figure(figsize=(10, 6))
        plt.plot(self.data)
        plt.xlabel("Episode")
        plt.ylabel("Average Reward")
        plt.title("Average Reward vs Episode")
        plt.show()

def run_as_player():
    env = Environment()
    done = False
    while not done:
        state, rewards, done = env.step()

def train_on_game():
    # Initialize environment and agents
    brain = Brain(model="DQN")
    env = Environment(brain=brain, max_time = 1024, tile_matrix=np.array([[0,0,0,0,0,0,0,0,0,0,0,0,0],
                                                                          [0,1,0,1,1,1,0,1,1,1,0,0,0],
                                                                          [0,0,0,0,0,0,0,0,0,0,0,0,0],
                                                                          [1,1,1,1,0,1,0,0,0,1,0,0,0],
                                                                          [1,0,0,0,0,1,0,1,0,0,0,0,0],
                                                                          [0,0,1,1,1,1,0,1,0,0,0,0,0],
                                                                          [0,0,0,0,0,0,0,0,0,0,0,0,0],
                                                                          [1,1,1,0,0,1,1,1,1,1,0,0,0],
                                                                          [0,0,0,2,0,0,0,1,0,0,0,0,0],
                                                                          [0,1,0,0,0,0,0,0,0,0,0,0,0]]))

    # Training settings
    num_episodes = 500
    batch_size = 128

    epsilon_start = 0.95
    epsilon_final = 0.0001
    epsilon_decay = 3000
    global_step = 0 
    epsilon = epsilon_final + (epsilon_start - epsilon_final) * math.exp(-1. * global_step / epsilon_decay)


    full_rewards = [0] #used for getting average rewards
    average_rewards = []
    times_explored = 0
    memory = ReplayBuffer(8192)
    display_freq=20
    done = False

    for episode in range(num_episodes):
        # Initialize replay memory

        if episode % display_freq == 0:
            env.reset(display=True, rand_start=False)
        else:
            env.reset(display=False, rand_start=False)

        
        average_reward = sum(full_rewards)/len(full_rewards)
        print(f'Finished: {done}, Average rewards for {episode}: {round(average_reward, 2)}, Time explored: {times_explored}')
        done = False
        average_rewards.append(average_reward)
        full_rewards = [] #used for getting average rewards
        times_explored = 0

        while env.time < 1024:  # Set a max number of steps per episode
            
            if episode % display_freq != 0 or episode == 0:
                explore = random.random() < epsilon
                env.actor.brain.init_explore()
            else:
                explore = False
            global_step += 1
            epsilon = epsilon_final + (epsilon_start - epsilon_final) * math.exp(-1. * global_step / epsilon_decay)
            for i in range(5):

                prev_state = env.get_state()
                #print(f'prev_state: {prev_state}')
                if explore:
                    times_explored += 1
                    actions = env.actor.use_brain(prev_state, explore=True)
                else:
                    output = env.actor.use_brain(prev_state)
                    #print(output)
                    actions = (output > 0.3)


                env.actor.apply_inputs(actions[0], actions[1], actions[2])
                state, rewards, done = env.step()
                full_rewards.append(rewards.item())
                

                memory.push(prev_state, actions, rewards, state, done)

                if done:
                    break
            if done:
                break

        if len(memory) > batch_size:
            for _ in range(3):
                env.actor.learn(memory, reward=average_reward)

            for _ in range(47):
                env.actor.learn(memory)

        # Debugging stats
        #grad_stats = get_gradient_stats(env.actor.brain.model)
        #print(f"Gradient stats: {grad_stats}")

    env.actor.brain.save_model("./Models", "model")
        
    # Plotting average rewards
    plt.figure(figsize=(10, 6))
    plt.plot(average_rewards)
    plt.xlabel("Episode")
    plt.ylabel("Average Reward")
    plt.savefig("average_reward_vs_episode.png")
    plt.title("Average Reward vs Episode")
    plt.show()


run_as_player()


