import torch
import torch.nn as nn
import torch.nn.functional as F

def fcLayer(input_units, output_units):
    return nn.Linear(input_units, output_units)

class ResBlock(nn.Module):
    '''
    Residual block of batch norm and fc layers
    '''
    def __init__(self, input_units, output_units, downsample=None):
        super(ResBlock, self).__init__()
        self.fc1 = fcLayer(input_units, output_units)
        self.bn1 = nn.BatchNorm1d(output_units)
        self.fc2 = fcLayer(output_units, output_units)
        self.bn2 = nn.BatchNorm1d(output_units)
        self.downsample = downsample
        
    def forward(self, x):
        residual = x
        out = self.fc1(x)
        out = F.relu(self.bn1(out))
        out = self.fc2(out)
        out = self.bn2(out)
        if self.downsample:
            residual = self.downsample(x)
        out += residual
        out = F.relu(out)
        return out
    
class QNetwork(nn.Module):
    '''
    Policy model
    '''
    def __init__(self, state_size, action_size, resblock, block_nums):
        super(QNetwork, self).__init__()
        self.input_units = state_size
        self.layer1 = self.make_layer(resblock, 512, block_nums[0])
        self.layer2 = self.make_layer(resblock, 256, block_nums[1])
        self.layer3 = self.make_layer(resblock, 128, block_nums[2])
        
        ## Dueling network
        # value stream
        self.value_fc = nn.Linear(128, 64)
        self.value = nn.Linear(64, 1)
        
        # advantage stream
        self.advantage_fc = nn.Linear(128, 64)
        self.advantage = nn.Linear(64, action_size)
        
    def make_layer(self, resblock, outputs, block_num):
        downsample = None
        if (self.input_units != outputs):
            downsample = nn.Sequential(fcLayer(self.input_units, outputs),
                                       nn.BatchNorm1d(outputs))
        layers = []
        layers.append(resblock(self.input_units, outputs, downsample))
        self.input_units = outputs
        for i in range(1, block_num):
            layers.append(resblock(outputs, outputs))
        return nn.Sequential(*layers)    
        
    def forward(self, state):
        '''
        definite forward pass
        '''
        x = self.layer1(state)
        x = self.layer2(x)
        x = self.layer3(x)
        
        value = F.relu(self.value_fc(x))
        value = self.value(value)
        
        advantage = F.relu(self.advantage_fc(x))
        advantage = self.advantage(advantage)
        
        return value + advantage - advantage.mean()