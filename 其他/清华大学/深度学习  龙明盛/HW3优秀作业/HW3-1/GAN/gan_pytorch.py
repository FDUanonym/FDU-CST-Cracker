import numpy as np

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as T
import torch.optim as optim
from torch.utils.data import sampler
import matplotlib.pyplot as plt


import PIL

NOISE_DIM = 96

dtype = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor

def sample_noise(batch_size, dim, seed=None):
    """
    Generate a PyTorch Tensor of uniform random noise.

    Input:
    - batch_size: Integer giving the batch size of noise to generate.
    - dim: Integer giving the dimension of noise to generate.

    Output:
    - A PyTorch Tensor of shape (batch_size, dim) containing uniform
      random noise in the range (-1, 1).
    """
    if seed is not None:
        torch.manual_seed(seed)
    ##############################################################################
    # TODO: generate noise                                                       #
    random_noise = torch.rand(batch_size, dim)
    random_noise = random_noise * 2 - 1
    return random_noise
    #                                                                            #
    ##############################################################################


def discriminator(seed=None):
    """
    Build and return a PyTorch model implementing the architecture above.
    """

    if seed is not None:
        torch.manual_seed(seed)

    ##############################################################################
    # TODO: Implement architecture                                               #
    #                                                                            #
    # HINT: nn.Sequential might be helpful. You'll start by calling Flatten().   #
    ##############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    class Discriminator(nn.Module):
        def __init__(self):
            super(Discriminator, self).__init__()
            self.main = nn.Sequential(
                nn.Linear(28 * 28, 512),
                nn.LeakyReLU(),
                nn.Linear(512, 256),
                nn.LeakyReLU(),
                nn.Linear(256, 1),
            )
            for m in self.modules():
                initialize_weights(m)

        def forward(self, x):
            x = x.view(-1, 28 * 28)
            x = self.main(x)
            return x

    model = Discriminator()

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ##############################################################################
    #                               END OF YOUR CODE                             #
    ##############################################################################
    return model


def generator(noise_dim=NOISE_DIM, seed=None):
    """
    Build and return a PyTorch model implementing the architecture above.
    """

    if seed is not None:
        torch.manual_seed(seed)

    ##############################################################################
    # TODO: Implement architecture                                               #

    class Generator(nn.Module):
        def __init__(self, noise_dim):
            super(Generator, self).__init__()
            self.main = nn.Sequential(
                nn.Linear(noise_dim, 256),
                nn.ReLU(),
                nn.Linear(256, 512),
                nn.ReLU(),
                nn.Linear(512, 28 * 28),
                nn.Tanh()
            )
            for m in self.modules():
                initialize_weights(m)

        def forward(self, x):
            img = self.main(x)
            return img

    model = Generator(noise_dim)

    # HINT: nn.Sequential might be helpful.                                      #
    ##############################################################################

    return model


def bce_loss(input, target):
    """
    Numerically stable version of the binary cross-entropy loss function in PyTorch.

    Inputs:
    - input: PyTorch Tensor of shape (N, ) giving scores.
    - target: PyTorch Tensor of shape (N,) containing 0 and 1 giving targets.

    Returns:
    - A PyTorch Tensor containing the mean BCE loss over the minibatch of input data.
    """
    bce = nn.BCEWithLogitsLoss()
    return bce(input.squeeze(), target.squeeze())

def discriminator_loss(logits_real, logits_fake):
    """
    Computes the discriminator loss described above.

    Inputs:
    - logits_real: PyTorch Tensor of shape (N,) giving scores for the real data.
    - logits_fake: PyTorch Tensor of shape (N,) giving scores for the fake data.

    Returns:
    - loss: PyTorch Tensor containing (scalar) the loss for the discriminator.
    """
    ##############################################################################
    # TODO:                                                                      #

    loss_real = bce_loss(logits_real, torch.ones_like(logits_real))
    loss_fake = bce_loss(logits_fake, torch.zeros_like(logits_fake))
    loss = loss_real + loss_fake

    #                                                                            #
    ##############################################################################
    return loss

def generator_loss(logits_fake):
    """
    Computes the generator loss described above.

    Inputs:
    - logits_fake: PyTorch Tensor of shape (N,) giving scores for the fake data.

    Returns:
    - loss: PyTorch Tensor containing the (scalar) loss for the generator.
    """
    ##############################################################################
    # TODO:                                                                      #

    loss = bce_loss(logits_fake, torch.ones_like(logits_fake))

    #                                                                            #
    ##############################################################################
    return loss

def get_optimizer(model):
    """
    Construct and return an Adam optimizer for the model with learning rate 1e-3,
    beta1=0.5, and beta2=0.999.

    Input:
    - model: A PyTorch model that we want to optimize.

    Returns:9
    - An Adam optimizer for the model with the desired hyperparameters.
    """
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    optimizer = optim.Adam(model.parameters(), lr=0.001, betas=(0.5, 0.999))

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    return optimizer

def ls_discriminator_loss(scores_real, scores_fake):
    """
    Compute the Least-Squares GAN loss for the discriminator.

    Inputs:
    - scores_real: PyTorch Tensor of shape (N,) giving scores for the real data.
    - scores_fake: PyTorch Tensor of shape (N,) giving scores for the fake data.

    Outputs:
    - loss: A PyTorch Tensor containing the loss.
    """
    ##############################################################################
    # TODO:                                                                      #

    loss_fn = nn.MSELoss()
    loss_real = loss_fn(torch.sigmoid(scores_real), torch.ones_like(scores_real)) / 2
    loss_fake = loss_fn(torch.sigmoid(scores_fake), torch.zeros_like(scores_fake)) / 2
    loss = loss_real + loss_fake

    #                                                                            #
    ##############################################################################
    return loss

def ls_generator_loss(scores_fake):
    """
    Computes the Least-Squares GAN loss for the generator.

    Inputs:
    - scores_fake: PyTorch Tensor of shape (N,) giving scores for the fake data.

    Outputs:
    - loss: A PyTorch Tensor containing the loss.
    """
    loss = None
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    loss_fn = nn.MSELoss()
    loss = loss_fn(torch.sigmoid(scores_fake), torch.ones_like(scores_fake)) / 2

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    return loss

def build_dc_classifier():
    """
    Build and return a PyTorch model for the DCGAN discriminator implementing
    the architecture above.
    """
    ##############################################################################
    # TODO: Implement architecture                                               #

    class Discriminator(nn.Module):  # input:1,28,28
        def __init__(self):
            super(Discriminator, self).__init__()
            self.conv = nn.Sequential(
                self.make_conv_block(1, 64),  # 64,13,13
                self.make_conv_block(64, 128, stride=2),  # 128,3,3
            )
            self.mlp = nn.Sequential(
                nn.Linear(128*3*3, 128),
                nn.LeakyReLU(),
                nn.Linear(128, 1),
            )
            for m in self.modules():
                initialize_weights(m)

        def make_conv_block(self, input_channels, output_channels, kernel_size=3, stride=1):
            return nn.Sequential(
                nn.Conv2d(input_channels, output_channels, kernel_size, stride),
                nn.LeakyReLU(0.2),
                nn.AvgPool2d(kernel_size=2, stride=2),
            )

        def forward(self, x):
            x = self.conv(x)
            x = x.view(-1, 128*3*3)
            x = self.mlp(x)
            return x

    model = Discriminator()
    return model

    # HINT: nn.Sequential might be helpful.                                      #
    ##############################################################################


def build_dc_generator(noise_dim=NOISE_DIM):
    """
    Build and return a PyTorch model implementing the DCGAN generator using
    the architecture described above.
    """

    ##############################################################################
    # TODO: Implement architecture                                               #

    class Generator(nn.Module):
        def __init__(self, noise_dim):
            super(Generator, self).__init__()
            self.mlp = nn.Sequential(
                self.make_mlp_block(noise_dim, 128),
                self.make_mlp_block(128, 128*7*7),
            )

            self.conv = nn.Sequential(
                nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),  # 64,14,14
                nn.ReLU(),
                nn.BatchNorm2d(64),
                nn.ConvTranspose2d(64, 1, kernel_size=4, stride=2, padding=1),  # 1,28,28
                nn.Tanh()
            )

            for m in self.modules():
                initialize_weights(m)

        def make_mlp_block(self, input_channels, output_channels):
            return nn.Sequential(
                nn.Linear(input_channels, output_channels),
                nn.ReLU(),
                nn.BatchNorm1d(output_channels),
            )

        def forward(self, x):
            x = self.mlp(x)
            x = x.view(-1, 128, 7, 7)
            img = self.conv(x)
            return img

    model = Generator(noise_dim)
    return model

    # HINT: nn.Sequential might be helpful.                                      #
    ##############################################################################


def run_a_gan(model_name, D, G, D_solver, G_solver, discriminator_loss, generator_loss, loader_train, show_every=250,
              batch_size=128, noise_size=96, num_epochs=10):
    """
    Train a GAN!

    Inputs:
    - D, G: PyTorch models for the discriminator and generator
    - D_solver, G_solver: torch.optim Optimizers to use for training the
      discriminator and generator.
    - discriminator_loss, generator_loss: Functions to use for computing the generator and
      discriminator loss, respectively.
    - show_every: Show samples after every show_every iterations.
    - batch_size: Batch size to use for training.
    - noise_size: Dimension of the noise to use as input to the generator.
    - num_epochs: Number of epochs over the training dataset to use for training.

    Outputs:
    - images: a list of image batch, each element has shape (batch_size, *figure_size)
    """
    images = []
    iter_count = 0
    D_loss = []
    G_loss = []
    for epoch in range(num_epochs):
        print('*' * 100)
        d_epoch_loss = 0
        g_epoch_loss = 0
        for x, _ in loader_train:
            if len(x) != batch_size:
                continue
            D_solver.zero_grad()
            real_data = x.type(dtype)
            logits_real = D(2 * (real_data - 0.5)).type(dtype)

            g_fake_seed = sample_noise(batch_size, noise_size).type(dtype)
            fake_images = G(g_fake_seed).detach()
            logits_fake = D(fake_images.view(batch_size, 1, 28, 28))

            d_total_error = discriminator_loss(logits_real, logits_fake)
            d_total_error.backward()
            D_solver.step()

            G_solver.zero_grad()
            g_fake_seed = sample_noise(batch_size, noise_size).type(dtype)
            fake_images = G(g_fake_seed)

            gen_logits_fake = D(fake_images.view(batch_size, 1, 28, 28))
            g_error = generator_loss(gen_logits_fake)
            g_error.backward()
            G_solver.step()

            if (iter_count % show_every == 0):
                print('Iter: {}, D: {:.4}, G:{:.4}'.format(iter_count, d_total_error.item(), g_error.item()))
                imgs_numpy = fake_images.data.cpu().numpy()
                images.append(imgs_numpy[0:16])

            with torch.no_grad():
                d_epoch_loss += d_total_error
                g_epoch_loss += g_error

            iter_count += 1

        with torch.no_grad():
            d_epoch_loss /= iter_count/(epoch+1)
            g_epoch_loss /= iter_count/(epoch+1)
            D_loss.append(d_epoch_loss.item())
            G_loss.append(g_epoch_loss.item())
            print('Epoch: {:d}/{:d}, D: {:.4}, G:{:.4}'.format(epoch+1, num_epochs, d_epoch_loss, g_epoch_loss))

    plt.figure()
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.plot(range(num_epochs), D_loss)
    plt.plot(range(num_epochs), G_loss)
    plt.legend(labels=['Discrimination', 'Generator'], loc='best')
    plt.savefig('./checkpoints/{}.png'.format(model_name))
    plt.close()

    return images



class ChunkSampler(sampler.Sampler):
    """Samples elements sequentially from some offset.
    Arguments:
        num_samples: # of desired datapoints
        start: offset where we should start selecting from
    """
    def __init__(self, num_samples, start=0):
        self.num_samples = num_samples
        self.start = start

    def __iter__(self):
        return iter(range(self.start, self.start + self.num_samples))

    def __len__(self):
        return self.num_samples


class Flatten(nn.Module):
    def forward(self, x):
        N, C, H, W = x.size() # read in N, C, H, W
        return x.view(N, -1)  # "flatten" the C * H * W values into a single vector per image

class Unflatten(nn.Module):
    """
    An Unflatten module receives an input of shape (N, C*H*W) and reshapes it
    to produce an output of shape (N, C, H, W).
    """
    def __init__(self, N=-1, C=128, H=7, W=7):
        super(Unflatten, self).__init__()
        self.N = N
        self.C = C
        self.H = H
        self.W = W
    def forward(self, x):
        return x.view(self.N, self.C, self.H, self.W)

def initialize_weights(m):
    if isinstance(m, nn.Linear) or isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Conv2d):
        nn.init.xavier_uniform_(m.weight.data)

def preprocess_img(x):
    return 2 * x - 1.0

def deprocess_img(x):
    return (x + 1.0) / 2.0

def rel_error(x,y):
    return np.max(np.abs(x - y) / (np.maximum(1e-8, np.abs(x) + np.abs(y))))

def count_params(model):
    """Count the number of parameters in the model. """
    param_count = np.sum([np.prod(p.size()) for p in model.parameters()])
    return param_count
