from itertools import chain

import torch
from torch import nn

from domain_shift.core.config import settings


class ResidualBlock(nn.Module):
    def __init__(self, in_features):
        super(ResidualBlock, self).__init__()
        self.block = nn.Sequential(
            nn.ReflectionPad1d(1),
            nn.Conv1d(in_features, in_features, kernel_size=3),
            nn.InstanceNorm1d(in_features),
            nn.ReLU(inplace=True),
            nn.ReflectionPad1d(1),
            nn.Conv1d(in_features, in_features, kernel_size=3),
            nn.InstanceNorm1d(in_features),
        )

    def forward(self, x):
        return x + self.block(x)


class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            # Reflection padding to preserve sequence length.
            nn.ReflectionPad1d(3),  # Sequence length: 6000 (unchanged)
            # Convolution layer: maps 1 channel to 64 channels.
            nn.Conv1d(
                1, 64, kernel_size=7
            ),  # Sequence length: 6000 - 7 + 1 + 6 (due to padding) = 6000
            nn.InstanceNorm1d(64),
            nn.ReLU(inplace=True),
            # Downsampling: halve the length by stride 2.
            nn.Conv1d(
                64, 128, kernel_size=3, stride=2, padding=1
            ),  # Sequence length: (6000 - 3 + 2) // 2 + 1 = 3000
            nn.InstanceNorm1d(128),
            nn.ReLU(inplace=True),
            # Downsampling again.
            nn.Conv1d(
                128, 256, kernel_size=3, stride=2, padding=1
            ),  # Sequence length: (3000 - 3 + 2) // 2 + 1 = 1500
            nn.InstanceNorm1d(256),
            nn.ReLU(inplace=True),
            # Residual blocks (length remains unchanged).
            *[
                ResidualBlock(256) for _ in range(9)
            ],  # Sequence length: 1500 (unchanged)
            # Upsampling: double the length by stride 2.
            nn.ConvTranspose1d(
                256, 128, kernel_size=3, stride=2, padding=1, output_padding=1
            ),  # Sequence length: (1500 - 1) * 2 + 3 = 3000
            nn.InstanceNorm1d(128),
            nn.ReLU(inplace=True),
            # Upsampling again.
            nn.ConvTranspose1d(
                128, 64, kernel_size=3, stride=2, padding=1, output_padding=1
            ),  # Sequence length: (3000 - 1) * 2 + 3 = 6000
            nn.InstanceNorm1d(64),
            nn.ReLU(inplace=True),
            # Final convolution to map back to single channel.
            nn.ReflectionPad1d(3),  # Sequence length: 6000 (unchanged)
            nn.Conv1d(64, 1, kernel_size=7),  # Sequence length: 6000 - 7 + 1 + 6 = 6000
            nn.Sigmoid(),  # Output in range [0, 1] for normalized data
        )

    def forward(self, x):
        return self.model(x)


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            # First convolution: map 1 channel to 64 channels.
            nn.Conv1d(
                1, 64, kernel_size=4, stride=2, padding=1
            ),  # Sequence length: (6000 - 4 + 2) // 2 + 1 = 3000
            nn.LeakyReLU(0.2, inplace=True),
            # Second convolution: map 64 channels to 128.
            nn.Conv1d(
                64, 128, kernel_size=4, stride=2, padding=1
            ),  # Sequence length: (3000 - 4 + 2) // 2 + 1 = 1500
            nn.InstanceNorm1d(128),
            nn.LeakyReLU(0.2, inplace=True),
            # Third convolution: map 128 channels to 256.
            nn.Conv1d(
                128, 256, kernel_size=4, stride=2, padding=1
            ),  # Sequence length: (1500 - 4 + 2) // 2 + 1 = 750
            nn.InstanceNorm1d(256),
            nn.LeakyReLU(0.2, inplace=True),
            # Fourth convolution: map 256 channels to 512.
            nn.Conv1d(
                256, 512, kernel_size=4, stride=2, padding=1
            ),  # Sequence length: (750 - 4 + 2) // 2 + 1 = 375
            nn.InstanceNorm1d(512),
            nn.LeakyReLU(0.2, inplace=True),
            # Final convolution: map 512 channels to 1 (for real/fake score).
            nn.Conv1d(512, 1, kernel_size=4),  # Sequence length: 375 - 4 + 1 = 372
            nn.AdaptiveAvgPool1d(1),  # Sequence length: 1
        )

    def forward(self, x):
        return self.model(x)


class CycleGAN:
    def __init__(self):
        # Initialize the device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Initialize the generators and discriminators.
        self.generator_1_to_2 = Generator().to(self.device)
        self.generator_2_to_1 = Generator().to(self.device)
        self.discriminator_1 = Discriminator().to(self.device)
        self.discriminator_2 = Discriminator().to(self.device)

        # Initialize the optimizers.
        self.optimizer_G = torch.optim.Adam(
            chain(
                self.generator_1_to_2.parameters(), self.generator_2_to_1.parameters()
            ),
            lr=settings.LR,
            betas=(0.5, 0.999),
        )
        self.optimizer_D = torch.optim.Adam(
            chain(self.discriminator_1.parameters(), self.discriminator_2.parameters()),
            lr=settings.LR,
            betas=(0.5, 0.999),
        )

        # Initialize the loss functions.
        self.criterion_GAN = torch.nn.MSELoss().to(self.device)
        self.criterion_cycle = torch.nn.L1Loss().to(self.device)
        self.criterion_identity = torch.nn.L1Loss().to(self.device)

    def train(self, data_loader_1, data_loader_2):
        # Set the models to training mode.
        self.generator_1_to_2.train()
        self.generator_2_to_1.train()
        self.discriminator_1.train()
        self.discriminator_2.train()

        # Train the models.
        for epoch in range(settings.EPOCHS):
            for i, (real_1, real_2) in enumerate(zip(data_loader_1, data_loader_2)):
                # Move data to the device.
                real_1 = real_1.to(self.device)
                real_2 = real_2.to(self.device)

                # Set the labels for real and fake data.
                real_label = torch.full(
                    (real_1.size(0), 1, 1), settings.REAL_LABEL, device=self.device
                )
                fake_label = torch.full(
                    (real_1.size(0), 1, 1), settings.FAKE_LABEL, device=self.device
                )

                # Train the generators.
                self.optimizer_G.zero_grad()

                # Identity loss.
                loss_id_1 = self.criterion_identity(
                    self.generator_2_to_1(real_1), real_1
                )
                loss_id_2 = self.criterion_identity(
                    self.generator_1_to_2(real_2), real_2
                )

                # GAN loss.
                fake_2 = self.generator_1_to_2(real_1)
                loss_GAN_1_to_2 = self.criterion_GAN(
                    self.discriminator_2(fake_2), real_label
                )
                fake_1 = self.generator_2_to_1(real_2)
                loss_GAN_2_to_1 = self.criterion_GAN(
                    self.discriminator_1(fake_1), real_label
                )

                # Cycle loss.
                recovered_1 = self.generator_2_to_1(fake_2)
                loss_cycle_1_2_1 = self.criterion_cycle(recovered_1, real_1)
                recovered_2 = self.generator_1_to_2(fake_1)
                loss_cycle_2_1_2 = self.criterion_cycle(recovered_2, real_2)

                # Total loss.
                loss_G = (
                    loss_id_1
                    + loss_id_2
                    + loss_GAN_1_to_2
                    + loss_GAN_2_to_1
                    + loss_cycle_1_2_1
                    + loss_cycle_2_1_2
                )
                loss_G.backward()
                self.optimizer_G.step()

                # Train the discriminators.
                self.optimizer_D.zero_grad()

                # Discriminator 1 loss.
                loss_real = self.criterion_GAN(self.discriminator_1(real_1), real_label)
                fake_1 = self.generator_2_to_1(real_2)
                loss_fake = self.criterion_GAN(
                    self.discriminator_1(fake_1.detach()), fake_label
                )
                loss_D_1 = (loss_real + loss_fake) / 2
                loss_D_1.backward()

                # Discriminator 2 loss.
                loss_real = self.criterion_GAN(self.discriminator_2(real_2), real_label)
                fake_2 = self.generator_1_to_2(real_1)
                loss_fake = self.criterion_GAN(
                    self.discriminator_2(fake_2.detach()), fake_label
                )
                loss_D_2 = (loss_real + loss_fake) / 2
                loss_D_2.backward()

                self.optimizer_D.step()

                # Print the losses.
                print(
                    f"Epoch [{epoch}/{settings.EPOCHS}] Batch [{i}/{len(data_loader_1)}] "
                    f"Loss G: {loss_G.item():.4f}, Loss D 1: {loss_D_1.item():.4f}, Loss D 2: {loss_D_2.item():.4f}"
                )

    def generate(self, generator: Generator, data_loader):
        # Set the generator to evaluation mode.
        generator.eval()

        # Generate the data.
        for i, real in enumerate(data_loader):
            real = real.to(self.device)  # Move data to the device.
            with torch.no_grad():  # Disable gradient computation for inference.
                synthetic = generator(real)
            yield synthetic.cpu()  # Move generated data back to the CPU for further processing.

        # Set the generator back to training mode.
        generator.train()

    def save_models(self):
        torch.save(self.generator_1_to_2.state_dict(), settings.GENERATOR_1_TO_2_PATH)
        torch.save(self.generator_2_to_1.state_dict(), settings.GENERATOR_2_TO_1_PATH)
        torch.save(self.discriminator_1.state_dict(), settings.DISCRIMINATOR_1_PATH)
        torch.save(self.discriminator_2.state_dict(), settings.DISCRIMINATOR_2_PATH)

    def load_models(self):
        self.generator_1_to_2.load_state_dict(
            torch.load(settings.GENERATOR_1_TO_2_PATH)
        )
        self.generator_2_to_1.load_state_dict(
            torch.load(settings.GENERATOR_2_TO_1_PATH)
        )
        self.discriminator_1.load_state_dict(torch.load(settings.DISCRIMINATOR_1_PATH))
        self.discriminator_2.load_state_dict(torch.load(settings.DISCRIMINATOR_2_PATH))
