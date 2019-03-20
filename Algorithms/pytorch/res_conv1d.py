import torch
import torch.nn as nn
import torch.nn.functional as F

def conv_2(in_channels, out_channels, stride=1):
    return nn.Sequential(nn.ConstantPad1d((0,1), 0), nn.Conv1d(in_channels, 
                         out_channels, kernel_size=2, stride=stride, padding=0, bias=False))

class ResBlock(nn.Module):
    '''
    Residual block of batch norm and fc layers
    '''
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(ResBlock, self).__init__()
        self.conv1 = conv_2(in_channels, out_channels)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = conv_2(out_channels, out_channels)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.downsample = downsample
        
    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = F.relu(self.bn2(out))
        out = self.conv2(out)
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
        self.in_channels = 16
        self.conv = conv_2(1, 16)
        self.bn = nn.BatchNorm1d(16)
        self.layer1 = self.make_layer(resblock, 16, block_nums[0])
        self.layer2 = self.make_layer(resblock, 32, block_nums[1])
        self.layer3 = self.make_layer(resblock, 64, block_nums[2])
        
        ## Dueling network
        # value stream
        self.value_fc = nn.Sequential(nn.Linear(state_size*64, 256),nn.Linear(256,64))
        self.value = nn.Linear(64, 1)
        
        # advantage stream
        self.advantage_fc = nn.Sequential(nn.Linear(state_size*64, 256),nn.Linear(256,64))
        self.advantage = nn.Linear(64, action_size)
        
    def make_layer(self, resblock, out_channels, block_num, stride=1):
        downsample = None
        if self.in_channels != out_channels:
            downsample = nn.Sequential(conv_2(self.in_channels, out_channels, stride=stride),
                                       nn.BatchNorm1d(out_channels))
        layers = []
        layers.append(resblock(self.in_channels, out_channels, stride, downsample))
        self.in_channels = out_channels
        for i in range(1, block_num):
            layers.append(resblock(out_channels, out_channels))
        return nn.Sequential(*layers)    
        
    def forward(self, state):
        '''
        definite forward pass
        '''
        x = F.relu(self.bn(self.conv(state)))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = x.view(x.size(0), -1)
        
        value = F.relu(self.value_fc(x))
        value = self.value(value)
        
        advantage = F.relu(self.advantage_fc(x))
        advantage = self.advantage(advantage)
        
        return value + advantage - advantage.mean()
