import torch
import torch.nn as nn
import torch.nn.functional as F

class QNetwork(nn.Module):
    '''
    Policy model
    '''
    def __init__(self, state_size, action_size):
        super(QNetwork, self).__init__()
        hidden_1 = 512
        hidden_2 = 512
        hidden_3 = 256
        hidden_4 = 256
        
        self.fc1 = nn.Linear(state_size, hidden_1)
        self.fc2 = nn.Linear(hidden_1, hidden_2)
        self.fc3 = nn.Linear(hidden_2, hidden_3)
        
        ## Dueling network
        # value stream
        self.value_fc = nn.Linear(hidden_3, hidden_4)
        self.value = nn.Linear(hidden_4, 1)
        
        # advantage stream
        self.advantage_fc = nn.Linear(hidden_3, hidden_4)
        self.advantage = nn.Linear(hidden_4, action_size)
        
        
    def forward(self, state):
        '''
        definite forward pass
        '''
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        
        value = F.relu(self.value_fc(x))
        value = self.value(value)
        
        advantage = F.relu(self.advantage_fc(x))
        advantage = self.advantage(advantage)
        
        return value + advantage - advantage.mean()