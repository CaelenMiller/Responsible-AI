
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import torch.optim as optim
from collections import deque
import os
import copy
import math
import numpy as np

from ModelClasses.KResNet import *
from ModelClasses.ModularNet import *


'''Rather than doing a homebrewed implementation, this version of the Brain class utilizes RLlibs implementations'''




class Brain:
    def __init__(self, learning_method = "DQN", optimizer_cls=optim.Adam, lr=0.001, batch_size=64, network_type="modular",
                 network_params = {"actor_state_size":8, "env_state_size":7*7, "output_size":3, "spatial":True,
                                   "nonspatial_depth":12, "nonspatial_width":128, "output_type":"distribution"}):

        self.learning_method = learning_method
        self.batch_size=batch_size
        self.lr = lr
        self.params = network_params
        self.ACTOR_STATE_SIZE = network_params["actor_state_size"]
        self.ENV_STATE_SIZE = network_params["env_state_size"]
        self.OUTPUT_SIZE = network_params["output_size"]
        self.SPATIAL = network_params["spatial"]
        self.NONSPATIAL_DEPTH = network_params["nonspatial_depth"]
        self.NONSPATIAL_WIDTH = network_params["nonspatial_width"]

        if network_type == "modular":
            self.net_class = Modular_Network
        elif network_type == "koopman":
            self.net_class = KoopmanResNet


        self.__init_learning__()
        
    def save_model(self, directory, filename):
        """
        Saves the model's state dictionary and the optimizer's state.

        Parameters:
        - directory (str): The path to the directory where to save the model.
        - filename (str): The name for the saved model file.
        """
        if not os.path.exists(directory):
            os.makedirs(directory)
        
        if self.learning_method == "DQN":
            file_path = os.path.join(directory, filename)
            torch.save({
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
            }, file_path)
        elif self.learning_method == "SAC":
            file_path_actor = os.path.join(directory, "actor_" + filename)
            file_path_critic = os.path.join(directory, "critic_" + filename)
            torch.save({
                'model_state_dict': self.actor.state_dict(),
                'optimizer_state_dict': self.actor_optimizer.state_dict(),
            }, file_path_actor)
            torch.save({
                'model_state_dict': self.critic.state_dict(),
                'optimizer_state_dict': self.critic_optimizer.state_dict(),
            }, file_path_critic)
        print(f"Model saved to {directory}/{filename}")

    def load_model(self, directory, filename, learning_method = "DQN", mode="train"):
        file_path = os.path.join(directory, filename)
        file_path_actor = os.path.join(directory, "actor_" + filename)

        if not os.path.exists(file_path) and not os.path.exists(file_path_actor):
            raise FileNotFoundError(f"No model file found at {file_path} or {file_path_actor}")

        #Add other models in here to allow them to be loaded
        if self.learning_method == "DQN":
            self.model = Modular_Network(self.ACTOR_STATE_SIZE, self.ENV_STATE_SIZE) 
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
            if mode == "train":
                self.model.train()
            else:
                self.model.eval() 

            checkpoint = torch.load(file_path)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

            self.target_model = copy.deepcopy(self.model)

        elif self.learning_method == "SAC":
            self.actor = Modular_Network(ACTOR_STATE_SIZE=self.ACTOR_STATE_SIZE, ENV_STATE_SIZE=self.ENV_STATE_SIZE, \
                             OUTPUT_SIZE=self.OUTPUT_SIZE, out_type="distribution", spatial_input=self.SPATIAL)
            self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=self.lr)

            #essentially a Q-net, maps from states + actions to expected rewards
            self.critic = DoubleQ(ACTOR_STATE_SIZE=self.ACTOR_STATE_SIZE + self.OUTPUT_SIZE, 
                                      ENV_STATE_SIZE=self.ENV_STATE_SIZE, spatial_input=self.SPATIAL)
            self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=self.lr)

            if mode == "train":
                self.actor.train()
                self.critic.train()
            else:
                self.actor.eval() 
                self.critic.eval()

            file_path_actor = os.path.join(directory, "actor_" + filename)
            file_path_critic = os.path.join(directory, "critic_" + filename)

            checkpoint = torch.load(file_path_actor)
            self.actor.load_state_dict(checkpoint['model_state_dict'])
            self.actor_optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            checkpoint = torch.load(file_path_critic)
            self.critic.load_state_dict(checkpoint['model_state_dict'])
            self.critic_optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

            self.target_critic = copy.deepcopy(self.critic)
                
        else:
            raise TypeError(f'Invalid learning method: {learning_method}')

        print(f"model successfully loaded from: {file_path}")

    def activate(self, inputs, deterministic=False):
        #with torch.no_grad:
            if self.learning_method == "DQN":
                outputs = self.model.forward(inputs.float().reshape(-1,self.ENV_STATE_SIZE+self.ACTOR_STATE_SIZE))
            elif self.learning_method == "SAC":
                outputs, _ = self.actor.forward(inputs.float().reshape(-1,self.ENV_STATE_SIZE+self.ACTOR_STATE_SIZE), deterministic=deterministic)
            else:
                raise TypeError(f'Invalid learning method: {self.learning_method}')
            return outputs
    
    def init_explore(self): #selects action to be taken until called again.
        self.explore_outputs = np.array(self.OUTPUT_SIZE)
        for output in range(len(self.explore_outputs)):
            #25% of 0, 25% chance of 1, smooth in between
            self.explore_outputs[output] = min(max((random.randint(0,200) - 50) / 100, 0), 1) 
        self.explore_outputs = torch.tensor(self.explore_outputs).reshape(1,self.OUTPUT_SIZE)
    
    def explore(self): #gets exploration outputs instead of running network
        return self.explore_outputs

    def __init_learning__(self): #initializes algorithm specific variables, for use in "self.learn"
        if self.learning_method == "DQN": 
            print("Initializing DQN")
            self.model = self.net_class(self.params)
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
            self.target_model = copy.deepcopy(self.model)
            self.timestep = 0
            self.TARGET_UPDATE_FREQUENCY = 500
            self.gamma = 0.98
            
        elif self.learning_method == "SAC":
            print("Initializing Soft Actor Critic")
            # actor outputs a distribution (mean and std dev for each action)
            self.actor = self.net_class(self.params)
            self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=self.lr)

            #essentially a Q-net, maps from states + actions to expected rewards
            q_params = copy.deepcopy(self.params)
            q_params["actor_state_size"] = self.params["actor_state_size"] + self.params["output_size"]
            q_params["output_type"] = "q-values"
            self.critic = DoubleQ(q_params, self.net_class)
            self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=self.lr)
            self.target_critic = copy.deepcopy(self.critic)

            self.gamma = 0.99
            self.alpha = 0.12  # Entropy coefficient
            self.tau = 0.005  # Soft update rate for target networks. 

            self.adaptive_alpha = True

            if self.adaptive_alpha:
                self.target_entropy = torch.tensor(-self.OUTPUT_SIZE, dtype=float, requires_grad=True)
                self.log_alpha = torch.tensor(np.log(self.alpha), dtype=float, requires_grad=True)
                self.alpha_optim = torch.optim.Adam([self.log_alpha], lr=self.lr)
            
        else:
            raise TypeError(f'Invalid learning algorithm: {self.learning_method}')


    def learn(self, memory, iterations): #runs learning algorithm of choice on current model
        losses = []
        stats = {"critic loss" : [], "actor loss" : [], "critic g-norm" : [], "actor g-norm" : [], "Q": [], "target Q": []}
        for i in range(iterations):
            if self.learning_method == "DQN":
                if self.timestep % self.TARGET_UPDATE_FREQUENCY == 0:
                    self.target_model.load_state_dict(self.model.state_dict())
                self.timestep += 1

                states, actions, rewards, next_states, _ = memory.sample(self.batch_size)
                states = torch.stack(states).reshape((-1, self.ENV_STATE_SIZE+self.ACTOR_STATE_SIZE)).float()
                actions = torch.stack(actions).reshape((-1, self.OUTPUT_SIZE)).float()
                rewards = torch.stack(rewards).reshape((-1, 1)).float()

                next_states = torch.stack(next_states).reshape((-1, self.ENV_STATE_SIZE+self.ACTOR_STATE_SIZE)).float()

                # Get the Q-values of the actions actually taken
                q_values = self.model(states) #Q(s,a)
                q_values = (q_values * actions).sum(dim=1).unsqueeze(1)
                
                # Compute the maximum Q-value, detach it from the graph to prevent backprop
                max_next_q_values = self.target_model(next_states).detach().max(1)[0].unsqueeze(1)

                # Compute the target Q-values
                next_q_values = rewards + (self.gamma * max_next_q_values)

                # Compute loss
                loss = F.mse_loss(q_values, next_q_values)
                
                # Zero gradients, perform a backward pass, and update the weights.
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                losses.append(loss.item())

            if self.learning_method == "SAC":
                # Sample a batch of transitions from memory
                states, actions, rewards, next_states, dones = memory.sample(self.batch_size)
            
                # Process states, actions, rewards, next_states, dones into correct size and datatype
                #print(states)
                states = torch.stack(states).reshape((-1, self.ENV_STATE_SIZE+self.ACTOR_STATE_SIZE)).float()
                actions = torch.stack(actions).reshape((-1, self.OUTPUT_SIZE)).float()
                rewards = torch.stack(rewards).reshape((-1, 1)).float()
                next_states = torch.stack(next_states).reshape((-1, self.ENV_STATE_SIZE+self.ACTOR_STATE_SIZE)).float()
                dones = torch.stack(dones).reshape((-1, 1)).float()
               
        
                # ---------------------- Update Critic ---------------------- #
                # Compute the target Q-values
                with torch.no_grad():
                    # Get next action probabilities and Q-values from target critic
                    next_actions, log_pi_a_next = self.actor(next_states)

                    target_Q1, target_Q2 = self.target_critic(torch.cat((next_states, next_actions), dim=1))
                    target_Q = torch.min(target_Q1, target_Q2)

                    # Compute the target Q-value using rewards and the next state's Q-value
                    # print(f'rewards: {rewards.shape}')
                    # print(f'dones: {dones.shape}')
                    # print(f'target_Q: {target_Q.shape}')
                    # print(f'log_pi_a_next: {log_pi_a_next.shape}')
                    target_Q = rewards + (1 - dones) * self.gamma * (target_Q - self.alpha * log_pi_a_next)
                
                # Get current Q-values from critic using current states and actions
                current_Q1, current_Q2 = self.critic(torch.cat((states, actions), dim=1))

                # Compute critic loss using MSE between current Q-values and target Q-values
                critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)

                # Zero gradients, backward pass on critic, and update critic parameters
                # ----------------------------------------------------------- #
                self.critic_optimizer.zero_grad()
                critic_loss.backward()
                #torch.nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=1.0) #prevent exploding gradients
                self.critic_optimizer.step()

                stats["critic loss"].append(critic_loss.item())
                stats["critic g-norm"].append(get_gradient_stats(self.critic)["norm"])


                # ---------------------- Update Actor ----------------------- #
                # Compute actor loss (usually by using the critic's Q-values)
                # Zero gradients, backward pass on actor, and update actor parameters
                # ----------------------------------------------------------- #
                actions, log_probs = self.actor(states)
                with torch.no_grad():
                    Q1, Q2 = self.critic(torch.cat((states, actions), dim=1))
                    Q = torch.min(Q1, Q2)

                actor_loss = (self.alpha * log_probs - Q).mean()
                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                #torch.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=1.0) #prevent exploding gradients
                self.actor_optimizer.step()


                stats["actor loss"].append(actor_loss.item())
                stats["actor g-norm"].append(get_gradient_stats(self.actor)["norm"])
                stats["Q"].append(torch.mean(Q).item())
                stats["target Q"].append(torch.mean(target_Q).item())

                # ----------------- Update Target Networks ------------------ #
                # Soft update the target critic network
                # ----------------------------------------------------------- #
                # for param, target_param in zip(self.critic.parameters(), self.target_critic.parameters()):
                #     target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

                # Optionally, adjust alpha (entropy coefficient) here. This should ensure that we don't get a wacky alpha
                if self.adaptive_alpha:
                    # We learn log_alpha instead of alpha to ensure alpha>0
                    alpha_loss = -(self.log_alpha * (log_probs + self.target_entropy).detach()).mean()
                    self.alpha_optim.zero_grad()
                    alpha_loss.backward()
                    self.alpha_optim.step()
                    self.alpha = self.log_alpha.exp()
                    
                losses.append((actor_loss.item(), critic_loss.item()))        

        for key in stats.keys():
            print(f'{key} : {round(sum(stats[key])/len(stats[key]), 3)}')

        return losses
    

class DoubleQ(nn.Module):
    def __init__(self, params, model_class):
        super(DoubleQ, self).__init__()

        #outsize should be 1, out_type is q_value
        self.Q1 = model_class(params) 
        self.Q2 = model_class(params)

    def forward(self, sa): #input should be state action pair. eg - sa = torch.cat([state, action], 1)
        q1 = self.Q1(sa)
        q2 = self.Q2(sa)
        return q1, q2

'''A replay buffer for use with DQN and SAC. Usage example is inside of main'''
class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        experience = (state, action, reward, next_state, done)
        self.buffer.append(experience)

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = zip(*batch)
        return state, action, reward, next_state, done

    def __len__(self):
        return len(self.buffer)
    
'''Tool for making sure gradients aren't going bananas'''
def get_gradient_stats(model):
    grad_stats = {'mean': [], 'max': [], 'min': [], 'std': [], 'norm': []}
    
    for name, param in model.named_parameters():
        if param.grad is not None:
            grad_stats['mean'].append(param.grad.data.mean())
            grad_stats['max'].append(param.grad.data.max())
            grad_stats['min'].append(param.grad.data.min())
            grad_stats['std'].append(param.grad.data.std())
            grad_stats['norm'].append(param.grad.data.norm(2))

    grad_stats['mean'] = round(torch.tensor(grad_stats['mean']).mean().item(),2)
    grad_stats['max'] =  round(torch.tensor(grad_stats['max']).max().item(),2)
    grad_stats['min'] =  round(torch.tensor(grad_stats['min']).min().item(),2)
    grad_stats['std'] =  round(torch.tensor(grad_stats['std']).mean().item(),2)
    grad_stats['norm'] = round(torch.tensor(grad_stats['norm']).sum().item(),2)
    
    return grad_stats

