import math
import torch 
from torch import nn
import torch.nn.init as init
from torch.distributions import Normal
import torch.nn.functional as F
import numpy as np


class Modular_Network(nn.Module):
    #actor_state_size and env_state_size can be seen as non-spatial and spatial inputs, respectively
    def __init__(self, params):
        super(Modular_Network, self).__init__()    

        self.params = params        
        self.out_type = self.params["output_type"]
        self.spatial_input = self.params["spatial"]
        self.ACTOR_STATE_SIZE = self.params["actor_state_size"]
        self.ENV_STATE_SIZE = self.params["env_state_size"]
        nonspatial_width = self.params["nonspatial_width"]
        nonspatial_depth = self.params["nonspatial_depth"]
        OUTPUT_SIZE = self.params["output_size"]

        # Hidden layers
        self.Nonspatial = nn.ModuleList([])
        for i in range(nonspatial_depth):
            self.Nonspatial.append(nn.Linear(nonspatial_width, nonspatial_width))
        
        # Convolutional layers
        if self.spatial_input:
            self.ENV_STATE_WIDTH = int(math.sqrt(self.ENV_STATE_SIZE))
            self.conv1 = nn.Conv2d(1, 64, kernel_size=3, stride=1)
            self.conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
            init.kaiming_normal_(self.conv1.weight, mode='fan_in', nonlinearity='relu')
            init.kaiming_normal_(self.conv2.weight, mode='fan_in', nonlinearity='relu')

            #combine spatial and non-spatial
            self.Nonspatial[0] = nn.Linear(self.ACTOR_STATE_SIZE + ((self.ENV_STATE_WIDTH-4)**2) * 64, nonspatial_width) 
        else:
            self.ENV_STATE_WIDTH = self.ENV_STATE_SIZE
            #no spatial, so just feed forward net
            self.Nonspatial[0] = nn.Linear(self.ACTOR_STATE_SIZE + self.ENV_STATE_SIZE, nonspatial_width)


        # Output layer(s)
        if self.out_type == "distribution": #Outputs a mean and std dev for each action
            self.mean_layer = nn.Linear(nonspatial_width, OUTPUT_SIZE)
            self.log_std_dev_layer = nn.Linear(nonspatial_width, OUTPUT_SIZE) #learn log_std_dev to keep it positive
            self.LOG_STD_MAX = 2
            self.LOG_STD_MIN = -20
        elif self.out_type == "actions": #Outputs actions directly
            self.action_layer = nn.Linear(nonspatial_width, OUTPUT_SIZE)
        elif self.out_type == "q-values": #Outputs q values
            self.action_layer = nn.Linear(nonspatial_width, 1)
        else:
            raise TypeError(f'Invalid output type: {self.out_type}')

    def forward(self, x, deterministic=False): #deterministic condition only used for "distribution" outputs
        # Separate the spatial and non-spatial components of the state

        if self.spatial_input:
            if x.shape[1] > self.ENV_STATE_WIDTH ** 2: #check if there is nonspatial input
                spatial_only = False
                spatial_state = x[:, :-self.ACTOR_STATE_SIZE].view(-1, 1, self.ENV_STATE_WIDTH, self.ENV_STATE_WIDTH).float()
                non_spatial_state = x[:, -self.ACTOR_STATE_SIZE:].float()
            else: #there is only spatial input
                spatial_only = True
                spatial_state = x.view(-1, 1, self.ENV_STATE_WIDTH, self.ENV_STATE_WIDTH).float()

            # Convolutional layers
            x = F.leaky_relu(self.conv1(spatial_state))
            x = F.leaky_relu(self.conv2(x))

            x = x.view(x.size(0), -1) 

            # Concatenate with non-spatial state
            if not spatial_only:
                x = torch.cat((x, non_spatial_state), dim=1)

        # Fully connected layers - beginning of network if no spatial input
        for layer in self.Nonspatial:
            x = F.leaky_relu(layer(x))


        # Output layer - differs depending on output type
        if self.out_type == "distribution": #Outputs an action and log probs for each action, between 0 and 1
            mean = self.mean_layer(x)       #finds mean and std dev, then samples
            if deterministic:
                return torch.sigmoid(mean), 0
            log_std_dev = self.log_std_dev_layer(x)
            log_std_dev = torch.clamp(log_std_dev, self.LOG_STD_MIN, self.LOG_STD_MAX) #log forces std_dev > 0
            std_dev = torch.exp(log_std_dev)
            distribution = Normal(mean, std_dev)

            action = distribution.rsample() #sample from out distribution
            action = torch.sigmoid(action)

            #reconstruct the log probs based on selected actions and distribution
            log_probs = distribution.log_prob(action).sum(axis=1, keepdim=True) - (2 * (np.log(2) - mean - F.softplus(-2 * mean))).sum(axis=1, keepdim=True)

            #action = torch.clamp(action, 0, 1) #might cause learning problems

            #print(f'mean: {torch.mean(mean)}, std_dev: {torch.mean(std_dev)}')

            return action, log_probs #actions and log probabilities of those actions being selected from distribution

        elif self.out_type == "actions": #Outputs actions directly, between 0 and 1
            action = self.action_layer(x)
            return torch.sigmoid(action) #Between 0 and 1
        
        elif self.out_type == "q-values":
            action = self.action_layer(x)
            return action

    
    def select_action(self, x):
        return self.forward(x)
    
    def get_info(self): #Implement this in a way that 
        return 
    
