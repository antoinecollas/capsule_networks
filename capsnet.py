import torch, sys, time
from torch import nn
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class MarginLoss(nn.Module):
    def __init__(self, m_plus=0.9, m_minus=0.1, weight=0.5):
        super(MarginLoss, self).__init__()
        self.m_plus, self.m_minus, self.weight = m_plus, m_minus, weight
    
    def forward(self, input, target):
        '''
            - input: [batch_size, nb_classes] each number is a probability (0<=p<=1)
            - target: [batch_size] each classe is an interger which belongs to [0, nb_classes[
            - output: scalar
        '''
        L = 0
        # we begin by computing the left part of the formula (eq 4.)
        zeros = torch.zeros(input.shape).to(device)
        m_plus = torch.zeros(input.shape).fill_(self.m_plus).to(device)
        loss = torch.max(zeros, m_plus-input)**2
        target_reshape = target.reshape((target.shape[0],1))
        mask = torch.zeros(input.shape).to(device).scatter_(1, target_reshape, 1)
        loss = mask*loss
        # then we compute the right part of the formula (eq 4.)
        zeros = torch.zeros(input.shape).to(device)
        m_minus = torch.zeros(input.shape).to(device).fill_(self.m_minus)
        loss_2 = torch.max(zeros, input-m_minus)**2
        mask = torch.ones(input.shape).to(device).scatter_(1, target_reshape, 0)
        weight = torch.zeros(loss_2.shape).to(device).fill_(self.weight)
        loss_2 = self.weight*mask*loss_2
        loss = loss + loss_2
        loss = loss.sum(dim=1)
        loss = loss.mean()
        return loss

def squash(s, dim=-1):
    squared_norm = (s ** 2).sum(dim=dim, keepdim=True)
    scale = squared_norm / (1 + squared_norm)
    return scale * s / torch.sqrt(squared_norm)

class PrimaryCaps(nn.Module):
    def __init__(self, kernel_size=9, in_channels=256, size_primary_caps=8, out_channels=32):
        super(PrimaryCaps, self).__init__()
        self.size_primary_caps, self.out_channels = size_primary_caps, out_channels
        self.conv2 = nn.Conv2d(in_channels=in_channels, out_channels=size_primary_caps*out_channels, kernel_size=9, stride=2, padding=0, dilation=1, groups=1, bias=True)

    def forward(self, x):
        '''
            Input: [batch_size, in_channels, in_height, in_width]
            Returns: [batch_size, out_channels, size_primary_caps, out_height, out_width]
        '''
        output = self.conv2(x)
        height, width = output.shape[-2:]
        output = output.reshape(-1, self.out_channels, self.size_primary_caps, height, width)
        output = squash(output, dim=-3)
        return output

def routing(u, nb_iterations=1):
    '''
        Input: [batch_size, out_height, in_caps, out_width]
        Returns: [batch_size, out_height, out_width]
    '''
    batch_size, out_height, in_caps, out_width = u.shape
    b = torch.zeros((batch_size, out_height, in_caps, 1)).to(device)
    for i in range(nb_iterations):
        c = torch.nn.functional.softmax(b, dim=-3)
        s = u*c
        s = torch.sum(s, dim=-2)
        v = squash(s, dim=-1)
        temp_v = v.reshape((batch_size, out_height, 1, out_width)).repeat((1, 1, in_caps, 1))
        agreement = torch.sum(u*temp_v, dim=-1)
        agreement = agreement.reshape((*agreement.shape, 1))
        b = b+agreement
    return v

class DigitCaps(nn.Module):
    def __init__(self, in_caps=1152, size_in_caps=8, in_channels=32, out_height=10, out_width=16, iter_routing=1):
        super(DigitCaps, self).__init__()
        self.W = nn.Parameter(torch.randn(1, out_height, in_caps, out_width, size_in_caps)) # default: [1, 10, 1152, 16, 8]
        self.iter = iter_routing

    def forward(self, x):
        '''
            Input: [batch_size, in_channels, size_primary_caps, in_height, in_width]
            Returns: [batch_size, out_height, out_width]
        '''
        batch_size, _, size_primary_caps, _, _ = x.shape
        output = x.reshape(batch_size, 1, -1, size_primary_caps, 1)
        output = self.W @ output
        output = output.reshape(output.shape[:-1])
        output = routing(output, self.iter)
        return output

class CapsNet(nn.Module):
    def __init__(self, kernel_size=9, channels_conv1=256, size_primary_caps=8, channels_conv2=32, size_digit_caps=16, nb_digits=10, iter_routing=3):
        super(CapsNet, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=channels_conv1, kernel_size=kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True),
            nn.ReLU()
            )
        self.primary_caps = PrimaryCaps(kernel_size=kernel_size, in_channels=channels_conv1, size_primary_caps=size_primary_caps, out_channels=channels_conv2)
        in_caps = 32*6*6 #1152 TODO: remove hard code
        self.digit_caps = DigitCaps(in_caps=in_caps, size_in_caps=size_primary_caps, in_channels=channels_conv2, out_height=nb_digits, out_width=size_digit_caps, iter_routing=iter_routing)

    def count_parameters(self):
            return sum(p.numel() for p in self.parameters() if p.requires_grad)
            
    def forward(self, input):
        output = self.conv1(input)
        output = self.primary_caps(output)
        output = self.digit_caps(output)
        output = torch.norm(output, p=2, dim=-1)
        return output

    # def extra_repr(self):
    #     # (Optional)Set the extra information about this module. You can test
    #     # it by printing an object of this class.
    #     return 'in_features={}, out_features={}, bias={}'.format(
    #         self.in_features, self.out_features, self.bias is not None
    #     )