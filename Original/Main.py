from Environment import *
from Brain import *

import random
import time
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
        state = env.get_state()

def train_on_game_modular(num_episodes = 30000, display_freq=100000, rand_start=True, learning_method="SAC", 
                  state_type="sensor", batch_size = 128):

    # Training settings
    global_step = 0


    # Initialize environment and agents
    brain = Brain(learning_method=learning_method, batch_size=batch_size, lr=0.00025,
                  network_params = {"actor_state_size":7, "env_state_size":7*7, "output_size":3, "spatial":True,
                                   "nonspatial_depth":12, "nonspatial_width":25, "output_type":"distribution"})
    #brain.load_model("./Models", "model")
    env = Environment(brain=brain, state_type = state_type, max_time = 2048,
                       tile_matrix=np.array([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                                             [1, 0, 0, 0, 0, 1, 1, 1, 1, 1],
                                             [1, 1, 1, 1, 0, 1, 1, 1, 1, 1],
                                             [1, 1, 1, 1, 0, 1, 1, 1, 1, 1],
                                             [1, 1, 1, 1, 0, 1, 1, 1, 1, 1],
                                             [1, 1, 1, 1, 0, 1, 1, 1, 1, 1],
                                             [1, 1, 1, 1, 0, 1, 1, 1, 1, 1],
                                             [1, 1, 1, 1, 0, 0, 0, 2, 1, 1],
                                             [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                                             [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]]))


    episode_rewards = [] #used for getting average rewards
    average_rewards = []
    dones = []
    end_dists = []
    actor_losses = []
    critic_losses = []
    memory = ReplayBuffer(32768)
    done = False

    for episode in range(1, num_episodes):
        dones.append(done)
        end_dists.append(env.map.get_smoothed_distance(int(env.actor.y),int(env.actor.x)))
        
        if episode > 50 and episode % 25 == 0:
            average_reward = sum(episode_rewards[-25:])/len(episode_rewards[-25:])
            average_rewards.append(average_reward)
            print(f'Episode: {episode} : Dones: {round(sum(dones[-25:])/25,3)}, AVG: {round(sum(episode_rewards[-25:])/25, 2)}, End dist: {round(sum(end_dists[-25:])/25,3)}')

        if episode % 5000 == 0:
            print("Generating new tilemap")
            new_tilemap = env.map.generate_random_tilemap(len(env.map.tile_matrix), len(env.map.tile_matrix[0]))
            env.map = Map(tile_matrix = new_tilemap)
            

        done = False
        episode_rewards = [] #used for getting average rewards

        #This is where instructions from the pause menu are dealt with. Press p or esc to open pause menu

        if env.get_pause_instruction() == "toggle_random_start":
            if rand_start:
                rand_start = False
            else:
                rand_start = True
            print(f"Random Start: {rand_start}")

        if env.get_pause_instruction() == "force_save":
            env.actor.brain.save_model("./Models", "model")

        if env.get_pause_instruction() == "end_training":
            print("Ending Training")
            break

        if env.get_pause_instruction() == "display_loss":
            plot("Actor Loss over time", "steps", "Log Loss", np.array(actor_losses))
            plot("Log Critic Loss over time", "steps", "Log Loss", np.log(np.array(critic_losses)))

        if env.get_pause_instruction() == "random_map":
            print("Randomizing Map")
            env.map.generate_map(env.map.generate_random_tilemap)

        if episode % display_freq == 0 or env.get_pause_instruction() == "force_display":
            env.reset(display=True, rand_start=rand_start)
        else:
            env.reset(display=False, rand_start=rand_start)

        env.reset_instructions() #instruction only executed once
        while env.time < env.max_time:  # Set a max number of steps per episode
            
            global_step += 1
            #explore = random.random() < epsilon
            #env.actor.brain.init_explore()
            #epsilon = epsilon_final + (epsilon_start - epsilon_final) * math.exp(-1. * global_step / epsilon_decay)
            for _ in range(25):
                state = env.get_state()


                with torch.no_grad():
                    actions = env.actor.use_brain(state)

                env.actor.apply_inputs(actions[0].item(), actions[1].item(), actions[2].item())
                next_state, rewards, done = env.step()
                episode_rewards.append(rewards.item())
                
                
                #print(f'State: {state.type}, actions: {actions.type}, Rewards: {rewards.type}, Next: {next_state.type}')
                memory.push(state, actions.clone().detach(), rewards, next_state, torch.tensor([done]))

                if done or env.time >= env.max_time:
                    break
            if done:
                break

        #plot("Episode Reward", "step", "Reward", episode_rewards)
        
    
        if episode % 50 == 0:
            if len(memory) > batch_size:
                losses = env.actor.learn(memory, 25)
                actor_loss = [loss[0] for loss in losses]
                critic_loss = [loss[1] for loss in losses]
                actor_losses.append(actor_loss)
                critic_losses.append(critic_loss)

        # Debugging stats
        #grad_stats = get_gradient_stats(env.actor.brain.model)
        #print(f"Gradient stats: {grad_stats}")

    env.actor.brain.save_model("./Models", "model")

    plot("Average Reward", "Episode", "Reward", average_rewards)

    plot("Log Actor Loss over time", "steps", "Log Loss", np.log(np.array(actor_losses)))

def train_on_game_resnet(num_episodes = 3000, display_freq=0, rand_start=False, learning_method="SAC", 
                  state_type="visual", batch_size = 128):

    # Training settings
    global_step = 0


    # Initialize environment and agents
    brain = Brain(learning_method=learning_method, batch_size=batch_size, lr=0.00025,
                  network_params = {"actor_state_size":7, "env_state_size":7*7, "output_size":3, "spatial":True,
                                   "nonspatial_depth":12, "nonspatial_width":256, "output_type":"distribution"})
    #brain.load_model("./Models", "model")
    env = Environment(brain=brain, state_type = state_type, max_time = 2048,
                       tile_matrix=np.array([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                                             [1, 0, 0, 0, 0, 1, 1, 1, 1, 1],
                                             [1, 1, 1, 1, 0, 1, 1, 1, 1, 1],
                                             [1, 1, 1, 1, 0, 1, 1, 1, 1, 1],
                                             [1, 1, 1, 1, 0, 1, 1, 1, 1, 1],
                                             [1, 1, 1, 1, 0, 1, 1, 1, 1, 1],
                                             [1, 1, 1, 1, 0, 1, 1, 1, 1, 1],
                                             [1, 1, 1, 1, 0, 0, 0, 2, 1, 1],
                                             [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                                             [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]]))


    episode_rewards = [] #used for getting average rewards
    average_rewards = []
    dones = []
    end_dists = []
    actor_losses = []
    critic_losses = []
    memory = ReplayBuffer(32768)
    done = False

    for episode in range(1, num_episodes):
        dones.append(done)
        end_dists.append(env.map.get_smoothed_distance(int(env.actor.y),int(env.actor.x)))
        
        if episode > 50 and episode % 25 == 0:
            average_reward = sum(episode_rewards[-25:])/len(episode_rewards[-25:])
            average_rewards.append(average_reward)
            print(f'Episode: {episode} : Dones: {round(sum(dones[-25:])/25,3)}, AVG: {round(sum(episode_rewards[-25:])/25, 2)}, End dist: {round(sum(end_dists[-25:])/25,3)}')

        if episode % 5000 == 0:
            print("Generating new tilemap")
            new_tilemap = env.map.generate_random_tilemap(len(env.map.tile_matrix), len(env.map.tile_matrix[0]))
            env.map = Map(tile_matrix = new_tilemap)
            

        done = False
        episode_rewards = [] #used for getting average rewards

        #This is where instructions from the pause menu are dealt with. Press p or esc to open pause menu

        if env.get_pause_instruction() == "toggle_random_start":
            if rand_start:
                rand_start = False
            else:
                rand_start = True
            print(f"Random Start: {rand_start}")

        if env.get_pause_instruction() == "force_save":
            env.actor.brain.save_model("./Models", "model")

        if env.get_pause_instruction() == "end_training":
            print("Ending Training")
            break

        if env.get_pause_instruction() == "display_loss":
            plot("Actor Loss over time", "steps", "Log Loss", np.array(actor_losses))
            plot("Log Critic Loss over time", "steps", "Log Loss", np.log(np.array(critic_losses)))

        if env.get_pause_instruction() == "random_map":
            print("Randomizing Map")
            env.map.generate_map(env.map.generate_random_tilemap)

        if episode % display_freq == 0 or env.get_pause_instruction() == "force_display":
            env.reset(display=True, rand_start=rand_start)
        else:
            env.reset(display=False, rand_start=rand_start)

        env.reset_instructions() #instruction only executed once
        while env.time < env.max_time:  # Set a max number of steps per episode
            
            global_step += 1
            #explore = random.random() < epsilon
            #env.actor.brain.init_explore()
            #epsilon = epsilon_final + (epsilon_start - epsilon_final) * math.exp(-1. * global_step / epsilon_decay)
            for _ in range(25):
                state = env.get_state()


                with torch.no_grad():
                    actions = env.actor.use_brain(state)

                env.actor.apply_inputs(actions[0].item(), actions[1].item(), actions[2].item())
                next_state, rewards, done = env.step()
                episode_rewards.append(rewards.item())
                
                
                #print(f'State: {state.type}, actions: {actions.type}, Rewards: {rewards.type}, Next: {next_state.type}')
                memory.push(state, actions.clone().detach(), rewards, next_state, torch.tensor([done]))

                if done or env.time >= env.max_time:
                    break
            if done:
                break

        #plot("Episode Reward", "step", "Reward", episode_rewards)
        
    
        if episode % 50 == 0:
            if len(memory) > batch_size:
                losses = env.actor.learn(memory, 25)
                actor_loss = [loss[0] for loss in losses]
                critic_loss = [loss[1] for loss in losses]
                actor_losses.append(actor_loss)
                critic_losses.append(critic_loss)

        # Debugging stats
        #grad_stats = get_gradient_stats(env.actor.brain.model)
        #print(f"Gradient stats: {grad_stats}")

    env.actor.brain.save_model("./Models", "model")

    plot("Average Reward", "Episode", "Reward", average_rewards)

    plot("Log Actor Loss over time", "steps", "Log Loss", np.log(np.array(actor_losses)))

def plot(name, x_lab, y_lab, data):
    plt.figure(figsize=(10, 6))
    plt.plot(data)
    plt.xlabel(x_lab)
    plt.ylabel(y_lab)
    #plt.savefig("average_reward_vs_episode.png")
    plt.title(name)
    plt.show()


#train_on_game_modular()
run_as_player()

