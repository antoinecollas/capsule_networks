import torch, sys, time, math
from torch import nn

class MarginLoss(nn.Module):
    def __init__(self, m_plus=0.9, m_minus=0.1, weight=0.5):
        super(MarginLoss, self).__init__()
        self.m_plus, self.m_minus, self.weight = m_plus, m_minus, weight
    
    def forward(self, target, input):
        '''
            - input: [batch_size, nb_classes] each number is a probability (0<=p<=1)
            - target: [batch_size] each classe is an interger which belongs to [0, nb_classes[
            - output: scalar
        '''
        # we begin by computing the left part of the formula (eq 4.)
        zeros = input.new_zeros(input.shape)
        m_plus = input.new_full(input.shape, self.m_plus)
        loss = torch.max(zeros, m_plus-input)**2
        target_reshape = target.reshape((target.shape[0],1))
        mask = input.new_zeros(input.shape).scatter_(1, target_reshape, 1)
        loss = mask*loss
        # then we compute the right part of the formula (eq 4.)
        zeros = input.new_zeros(input.shape)
        m_minus = input.new_full(input.shape, self.m_minus)
        loss_2 = torch.max(zeros, input-m_minus)**2
        mask = input.new_ones(input.shape).scatter_(1, target_reshape, 0)
        weight = input.new_full(loss_2.shape, self.weight)
        loss_2 = self.weight*mask*loss_2
        loss = loss + loss_2
        loss = loss.sum(dim=1)
        loss = loss.mean()
        return loss

class TotalLoss(nn.Module):
    def __init__(self, m_plus=0.9, m_minus=0.1, weight_margin=0.5, weight_mse=0.0005):
        super(TotalLoss, self).__init__()
        self.margin_loss = MarginLoss(m_plus, m_minus, weight_margin)
        self.weight_mse = weight_mse
        self.mse = nn.MSELoss()
    
    def forward(self, image, reconstructed_image, labels, prediction):
        '''
            - image
            - reconstructed_image: [batch_size, size_image]
            - labels: [batch_size] each classe is an interger which belongs to [0, nb_classes[
            - prediction
            - output: scalar
        '''
        batch_size = image.shape[0]
        image = image.reshape(batch_size, -1)
        loss = self.margin_loss(labels, prediction) + self.weight_mse*self.mse(image, reconstructed_image)
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
    b = u.new_zeros((batch_size, out_height, in_caps, 1))
    for i in range(nb_iterations):
        c = torch.nn.functional.softmax(b, dim=-3)
        s = u*c
        s = torch.sum(s, dim=-2)
        v = squash(s, dim=-1)
        temp_v = v.reshape((batch_size, out_height, 1, out_width)).repeat((1, 1, in_caps, 1))
        agreement = u*temp_v
        agreement = torch.sum(agreement, dim=-1)
        agreement = agreement.reshape((*agreement.shape, 1))
        b = b+agreement
    return v

class DigitCaps(nn.Module):
    def __init__(self, in_caps=1152, size_in_caps=8, in_channels=32, out_height=10, out_width=16, iter_routing=1):
        super(DigitCaps, self).__init__()
        self.W = nn.Parameter(torch.randn(1, out_height, in_caps, out_width, size_in_caps)) # default: [1, 10, 1152, 16, 8]
        self.iter = iter_routing
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.W.size(-1))
        self.W.data.uniform_(-stdv, stdv)

    def forward(self, x):
        '''
            Input: [batch_size, in_channels, size_primary_caps, in_height, in_width]
            Returns: [batch_size, out_height, out_width]
        '''
        batch_size, _, size_primary_caps, _, _ = x.shape
        output = x.reshape(batch_size, 1, -1, size_primary_caps, 1)
        output = self.W @ output
        output = output.squeeze()
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
            
    def forward(self, input):
        output = self.conv1(input)
        output = self.primary_caps(output)
        output = self.digit_caps(output)
        output_norm = torch.norm(output, p=2, dim=-1)
        return output_norm, output

class Decoder(nn.Module):
    def __init__(self, size_digit_caps=16, out_features=784):
        super(Decoder, self).__init__()
        self.decoder = nn.Sequential(
            nn.Linear(in_features=size_digit_caps, out_features=512),
            nn.ReLU(),
            nn.Linear(in_features=512, out_features=1024),
            nn.ReLU(),
            nn.Linear(in_features=1024, out_features=out_features),
            nn.Sigmoid()
        )

    def forward(self, input, digit=None):
        '''
            input: [batch_size, nb_digits, size_primary_caps]
            digit: [batch_size]
            returns: [batch_size, out_features]
        '''
        if not self.training:
            input_norm = torch.norm(input, p=2, dim=-1)
            _, digit = torch.max(input_norm, 1)
        if self.training and digit is None:
            raise "Error, there are no digits whereas Decoder is in training mode."
        digit = digit.reshape((digit.shape[0],1))
        mask = input.new_zeros(input.shape[:-1])
        mask = mask.scatter_(1, digit, 1)
        mask = mask.reshape((*mask.shape,1)).repeat((1,1,input.shape[-1]))
        input = input*mask
        input = input.sum(dim=-2)
        output = self.decoder(input)
        return output

class Model(nn.Module):
    def __init__(self, kernel_size=9, channels_conv1=256, size_primary_caps=8, channels_conv2=32, size_digit_caps=16, nb_digits=10, iter_routing=3, out_features=784):
        super(Model, self).__init__()
        self.capsnet = CapsNet(kernel_size, channels_conv1, size_primary_caps, channels_conv2, size_digit_caps, nb_digits, iter_routing)
        self.decoder = Decoder(size_digit_caps, out_features)

    def count_parameters(self):
            return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(self, input, labels=None):
        output_norm, output = self.capsnet(input)
        reconstructed_image = self.decoder(output, labels)
        return output_norm, reconstructed_image