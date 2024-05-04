import torch
from torch import nn
import torch.nn.functional as F

class Discriminator(torch.nn.Module):
    def __init__(self, input_channels=3):
        super(Discriminator, self).__init__()

        ####################################
        #          YOUR CODE HERE          #
        ####################################

        ##########       END      ##########
        self.conv1 = nn.Conv2d(input_channels, 128, 4, 2, 1)
        self.conv2 = nn.Conv2d(128, 256, 4, 2, 1)
        self.bn2 = nn.BatchNorm2d(256)
        self.conv3 = nn.Conv2d(256, 512, 4, 2, 1)
        self.bn3 = nn.BatchNorm2d(512)
        self.conv4 = nn.Conv2d(512, 1024, 4, 2, 1)
        self.bn4 = nn.BatchNorm2d(1024)
        self.conv5 = nn.Conv2d(1024, 1, 4, 1, 0)



    def forward(self, x):

        ####################################
        #          YOUR CODE HERE          #
        ####################################

        ##########       END      ##########

        x = F.leaky_relu(self.conv1(x), 0.2)
        x = self.bn2(F.leaky_relu(self.conv2(x), 0.2))
        x = self.bn3(F.leaky_relu(self.conv3(x), 0.2))
        x = self.bn4(F.leaky_relu(self.conv4(x), 0.2))
        x = self.conv5(x)

        # x = torch.sigmoid(self.conv5(x))
        # Flatten the output
        return x.view(-1, 1).squeeze(1)


class Generator(torch.nn.Module):
    def __init__(self, noise_dim, output_channels=3):
        super(Generator, self).__init__()
        self.noise_dim = noise_dim
        self.tconv1 = nn.ConvTranspose2d(noise_dim, 1024, 4, 1, 0)
        self.bn1 = nn.BatchNorm2d(1024)
        self.tconv2 = nn.ConvTranspose2d(1024, 512, 4, 2, 1)
        self.bn2 = nn.BatchNorm2d(512)
        self.tconv3 = nn.ConvTranspose2d(512, 256, 4, 2, 1)
        self.bn3 = nn.BatchNorm2d(256)
        self.tconv4 = nn.ConvTranspose2d(256, 128, 4, 2, 1)
        self.bn4 = nn.BatchNorm2d(128)
        self.tconv5 = nn.ConvTranspose2d(128, output_channels, 4, 2, 1)

        ####################################
        #          YOUR CODE HERE          #
        ####################################

        ##########       END      ##########

    def forward(self, x):

        ####################################
        #          YOUR CODE HERE          #
        ####################################

        ##########       END      ##########

        x = x.view(-1, self.noise_dim, 1, 1)
        x = self.bn1(F.relu(self.tconv1(x)))
        x = self.bn2(F.relu(self.tconv2(x)))
        x = self.bn3(F.relu(self.tconv3(x)))
        x = self.bn4(F.relu(self.tconv4(x)))
        x = torch.tanh(self.tconv5(x))
        return x

