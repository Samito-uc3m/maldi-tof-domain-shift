import time
from itertools import chain, cycle
from pathlib import Path

import numpy as np
import torch
from torch import nn

from domain_shift.core.config import settings


def zip_repeat(*iterables):
    max_len = max(len(it) for it in iterables)  # Get the max length
    iterators = [cycle(it) for it in iterables]  # Create cycling iterators

    for _ in range(max_len):  # Yield values one at a time, like zip()
        yield tuple(next(it) for it in iterators)


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
    def __init__(
        self,
        lambda_multiplier: float = settings.LAMBDA_MULTIPLIER,
        epochs: int = settings.EPOCHS,
        lr: float = settings.LR,
        patience: int = settings.PATIENCE,
        triplet_weight: float = settings.TRIPLET_WEIGHT,  # <-- Weight for triplet loss
        num_classes: int = settings.NUM_CLASSES,
    ):
        # Initialize device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.lambda_multiplier = lambda_multiplier
        self.epochs = epochs
        self.patience = patience
        self.triplet_weight = triplet_weight

        # Initialize generators and discriminators
        self.generator_1_to_2 = Generator().to(self.device)
        self.generator_2_to_1 = Generator().to(self.device)
        self.discriminator_1 = Discriminator().to(self.device)
        self.discriminator_2 = Discriminator().to(self.device)

        # Initialize optimizers
        self.optimizer_G = torch.optim.Adam(
            chain(
                self.generator_1_to_2.parameters(), self.generator_2_to_1.parameters()
            ),
            lr=lr,
            betas=(0.5, 0.999),
        )
        self.optimizer_D = torch.optim.Adam(
            chain(self.discriminator_1.parameters(), self.discriminator_2.parameters()),
            lr=lr,
            betas=(0.5, 0.999),
        )

        # Initialize loss functions
        self.criterion_GAN = torch.nn.MSELoss().to(self.device)
        self.criterion_cycle = torch.nn.L1Loss().to(self.device)
        self.criterion_identity = torch.nn.L1Loss().to(self.device)
        self.criterion_triplet = nn.TripletMarginLoss().to(self.device)

        # Error trackers
        self.gan_error_loss = []
        self.cycle_error_loss = []
        self.triplet_error_loss = []
        self.identity_error_loss = []

    def train(
        self,
        train_data_loader_1,
        train_data_loader_2,
        # val_data_loader_1,
        # val_data_loader_2,
    ):
        # Set models to training mode
        self.generator_1_to_2.train()
        self.generator_2_to_1.train()
        self.discriminator_1.train()
        self.discriminator_2.train()

        # Best validation loss
        current_patience = 0
        best_val_loss = np.inf

        # Begin training
        for epoch in range(self.epochs):
            gan_error_loss = 0.0
            cycle_error_loss = 0.0
            identity_error_loss = 0.0
            triplet_error_loss = 0.0

            # Time tracking
            start_time = time.time()

            for i, (
                (real_1_anchor, real_1_positive, real_1_negative),
                (real_2_anchor, real_2_positive, real_2_negative),
            ) in enumerate(zip_repeat(train_data_loader_1, train_data_loader_2)):
                # ---------------------
                # Move triplets to device
                # ---------------------
                real_1_anchor = real_1_anchor.to(self.device)
                real_1_positive = real_1_positive.to(self.device)
                real_1_negative = real_1_negative.to(self.device)

                real_2_anchor = real_2_anchor.to(self.device)
                real_2_positive = real_2_positive.to(self.device)
                real_2_negative = real_2_negative.to(self.device)

                # ---------------------
                # Labels for real/fake
                # ---------------------
                batch_size = real_1_anchor.size(0)
                real_label = torch.full(
                    (batch_size, 1, 1), settings.REAL_LABEL, device=self.device
                )
                fake_label = torch.full(
                    (batch_size, 1, 1), settings.FAKE_LABEL, device=self.device
                )

                # =================================================
                #              1) Train Generators
                # =================================================
                self.optimizer_G.zero_grad()

                #
                # --- a) Domain 1 -> Domain 2 ---
                #
                # We'll treat "anchor" as the main sample for cycle, but we
                # also forward positive/negative through the generator
                # for the triplet loss.

                # Fake from anchor
                fake_2_anchor = self.generator_1_to_2(real_1_anchor)
                # Fake from pos/neg
                fake_2_positive = self.generator_1_to_2(real_1_positive)
                fake_2_negative = self.generator_1_to_2(real_1_negative)

                # GAN loss for anchor
                loss_GAN_1_to_2 = self.criterion_GAN(
                    self.discriminator_2(fake_2_anchor), fake_label  # Is this correct?
                )

                # Cycle: anchor -> fake2 -> recovered1
                recovered_1_anchor = self.generator_2_to_1(fake_2_anchor)
                loss_cycle_1_2_1 = self.criterion_cycle(
                    recovered_1_anchor, real_1_anchor
                )

                # Identity loss.
                loss_id_1 = self.criterion_identity(
                    self.generator_2_to_1(real_1_anchor), real_1_anchor
                )

                #
                # --- b) Domain 2 -> Domain 1 ---
                #
                # Similarly for domain 2 anchor

                fake_1_anchor = self.generator_2_to_1(real_2_anchor)
                fake_1_positive = self.generator_2_to_1(real_2_positive)
                fake_1_negative = self.generator_2_to_1(real_2_negative)

                # GAN loss for anchor
                loss_GAN_2_to_1 = self.criterion_GAN(
                    self.discriminator_1(fake_1_anchor), fake_label  # Is this correct?
                )

                # Cycle: anchor -> fake1 -> recovered2
                recovered_2_anchor = self.generator_1_to_2(fake_1_anchor)
                loss_cycle_2_1_2 = self.criterion_cycle(
                    recovered_2_anchor, real_2_anchor
                )

                # Identity loss.
                loss_id_2 = self.criterion_identity(
                    self.generator_1_to_2(real_2_anchor), real_2_anchor
                )

                #
                # --- c) Triplet Losses (Domain 1 & Domain 2) ---
                #
                # Flatten or embed the generated images for anchor, positive, negative.
                # We'll compute a separate triplet loss for each domain.

                # Domain 1->2 triplet
                anchor_emb_1to2 = fake_2_anchor.view(batch_size, -1)
                positive_emb_1to2 = fake_2_positive.view(batch_size, -1)
                negative_emb_1to2 = fake_2_negative.view(batch_size, -1)
                loss_triplet_1to2 = self.criterion_triplet(
                    anchor_emb_1to2, positive_emb_1to2, negative_emb_1to2
                )

                # Domain 2->1 triplet
                anchor_emb_2to1 = fake_1_anchor.view(batch_size, -1)
                positive_emb_2to1 = fake_1_positive.view(batch_size, -1)
                negative_emb_2to1 = fake_1_negative.view(batch_size, -1)
                loss_triplet_2to1 = self.criterion_triplet(
                    anchor_emb_2to1, positive_emb_2to1, negative_emb_2to1
                )

                loss_triplet = loss_triplet_1to2 + loss_triplet_2to1

                #
                # --- d) Total generator loss ---
                #
                loss_G = (
                    loss_GAN_1_to_2
                    + loss_GAN_2_to_1
                    + self.lambda_multiplier
                    * (loss_cycle_1_2_1 + loss_cycle_2_1_2 + loss_id_1 + loss_id_2)
                    + self.triplet_weight * loss_triplet
                )

                loss_G.backward()
                self.optimizer_G.step()

                # Accumulate for logging
                gan_error_loss += loss_GAN_1_to_2.item() + loss_GAN_2_to_1.item()
                cycle_error_loss += loss_cycle_1_2_1.item() + loss_cycle_2_1_2.item()
                identity_error_loss += loss_id_1.item() + loss_id_2.item()
                triplet_error_loss += loss_triplet.item()

                # =================================================
                #          2) Train Discriminators
                # =================================================
                self.optimizer_D.zero_grad()

                #
                # --- Discriminator 1 (domain1) ---
                #
                # Real
                loss_real_d1 = self.criterion_GAN(
                    self.discriminator_1(real_1_anchor), real_label
                )
                # Fake
                fake_1_anchor_detached = fake_1_anchor.detach()
                loss_fake_d1 = self.criterion_GAN(
                    self.discriminator_1(fake_1_anchor_detached), fake_label
                )
                loss_D_1 = (loss_real_d1 + loss_fake_d1) / 2
                loss_D_1.backward()

                #
                # --- Discriminator 2 (domain2) ---
                #
                loss_real_d2 = self.criterion_GAN(
                    self.discriminator_2(real_2_anchor), real_label
                )
                fake_2_anchor_detached = fake_2_anchor.detach()
                loss_fake_d2 = self.criterion_GAN(
                    self.discriminator_2(fake_2_anchor_detached), fake_label
                )
                loss_D_2 = (loss_real_d2 + loss_fake_d2) / 2
                loss_D_2.backward()

                self.optimizer_D.step()

                # Optionally free up memory
                del (
                    real_1_anchor,
                    real_1_positive,
                    real_1_negative,
                    real_2_anchor,
                    real_2_positive,
                    real_2_negative,
                    fake_1_anchor,
                    fake_1_positive,
                    fake_1_negative,
                    fake_2_anchor,
                    fake_2_positive,
                    fake_2_negative,
                    recovered_1_anchor,
                    recovered_2_anchor,
                )

                # print(f"Runnning batch {i}/{max(len(train_data_loader_1), len(train_data_loader_2))}")

            # ---------------------------
            # Print epoch information
            # ---------------------------
            avg_gan_loss = gan_error_loss / len(train_data_loader_1)
            avg_cycle_loss = cycle_error_loss / len(train_data_loader_1)
            avg_triplet_loss = triplet_error_loss / len(train_data_loader_1)
            avg_identity_loss = identity_error_loss / len(train_data_loader_1)

            # End time
            end_time = time.time()

            print(
                f"Epoch {epoch + 1}/{self.epochs} in {end_time-start_time}s, "
                f"Loss G: {loss_G.item():.4f}, "
                f"Loss D1: {loss_D_1.item():.4f}, "
                f"Loss D2: {loss_D_2.item():.4f}, "
                f"Avg GAN: {avg_gan_loss:.4f}, "
                f"Avg Cycle: {avg_cycle_loss:.4f}, "
                f"Avg Identity: {avg_identity_loss:.4f}, "
                f"Triplet: {avg_triplet_loss:.4f}"
            )
            self.gan_error_loss.append(avg_gan_loss)
            self.cycle_error_loss.append(self.lambda_multiplier * avg_cycle_loss)
            self.identity_error_loss.append(self.lambda_multiplier * avg_identity_loss)
            self.triplet_error_loss.append(self.triplet_weight * avg_triplet_loss)

            # ---------------------------
            # Validate the models
            # ---------------------------
            # val_error_loss = self.validate(val_data_loader_1, val_data_loader_2)
            val_error_loss = (
                self.triplet_weight * triplet_error_loss
                + self.lambda_multiplier * avg_cycle_loss
            )
            if val_error_loss < best_val_loss:
                print("\tValidation loss improved, saving model.")
                current_patience = 0
                best_val_loss = val_error_loss
                self.save_checkpoint_models()
            else:
                current_patience += 1
                print(
                    f"\tValidation loss did not improve, patience: {current_patience}"
                )
                # if current_patience >= self.patience:
                #     print("\tEarly stopping triggered.")
                #     break

        # # End of training
        # print("Training complete, loading best model.")
        # self.load_checkpoint_models()

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

    def validate(self, val_data_loader_1, val_data_loader_2):
        # Set the models to evaluation mode.
        self.generator_1_to_2.eval()
        self.generator_2_to_1.eval()

        # Calculate identity loss.
        cycle_error_loss = 0

        with torch.no_grad():
            for real_1_anchor, real_1_positive, real_1_negative in val_data_loader_1:
                # Move data to the device.
                real_1_anchor = real_1_anchor.to(self.device)

                # Cycle loss.
                recovered_1 = self.generator_2_to_1(
                    self.generator_1_to_2(real_1_anchor)
                )
                loss_cycle_1 = self.criterion_cycle(recovered_1, real_1_anchor)
                cycle_error_loss += loss_cycle_1

                # Free up memory.
                del real_1_anchor, real_1_positive, real_1_negative, recovered_1

            for real_2_anchor, real_2_positive, real_2_negative in val_data_loader_2:
                # Move data to the device.
                real_2_anchor = real_2_anchor.to(self.device)

                # Cycle loss.
                recovered_2 = self.generator_1_to_2(
                    self.generator_2_to_1(real_2_anchor)
                )
                loss_cycle_2 = self.criterion_cycle(recovered_2, real_2_anchor)
                cycle_error_loss += loss_cycle_2

                # Free up memory.
                del real_2_anchor, real_2_positive, real_2_negative, recovered_2

        # Models to training mode.
        self.generator_1_to_2.train()
        self.generator_2_to_1.train()

        return self.lambda_multiplier * cycle_error_loss.cpu().detach().item()

    def save_models(self):
        torch.save(self.generator_1_to_2.state_dict(), settings.GENERATOR_1_TO_2_PATH)
        torch.save(self.generator_2_to_1.state_dict(), settings.GENERATOR_2_TO_1_PATH)
        torch.save(self.discriminator_1.state_dict(), settings.DISCRIMINATOR_1_PATH)
        torch.save(self.discriminator_2.state_dict(), settings.DISCRIMINATOR_2_PATH)

    def save_checkpoint_models(self):
        torch.save(
            self.generator_1_to_2.state_dict(), settings.TEMP_GENERATOR_1_TO_2_PATH
        )
        torch.save(
            self.generator_2_to_1.state_dict(), settings.TEMP_GENERATOR_2_TO_1_PATH
        )
        torch.save(
            self.discriminator_1.state_dict(), settings.TEMP_DISCRIMINATOR_1_PATH
        )
        torch.save(
            self.discriminator_2.state_dict(), settings.TEMP_DISCRIMINATOR_2_PATH
        )

    def load_models(self):
        self.generator_1_to_2.load_state_dict(
            torch.load(settings.GENERATOR_1_TO_2_PATH)
        )
        self.generator_2_to_1.load_state_dict(
            torch.load(settings.GENERATOR_2_TO_1_PATH)
        )
        self.discriminator_1.load_state_dict(torch.load(settings.DISCRIMINATOR_1_PATH))
        self.discriminator_2.load_state_dict(torch.load(settings.DISCRIMINATOR_2_PATH))

    def load_checkpoint_models(self):
        self.generator_1_to_2.load_state_dict(
            torch.load(settings.TEMP_GENERATOR_1_TO_2_PATH)
        )
        self.generator_2_to_1.load_state_dict(
            torch.load(settings.TEMP_GENERATOR_2_TO_1_PATH)
        )
        self.discriminator_1.load_state_dict(
            torch.load(settings.TEMP_DISCRIMINATOR_1_PATH)
        )
        self.discriminator_2.load_state_dict(
            torch.load(settings.TEMP_DISCRIMINATOR_2_PATH)
        )

    def load_models_via_paths(
        self,
        generator_1_to_2_path: Path = settings.GENERATOR_1_TO_2_PATH,
        generator_2_to_1_path: Path = settings.GENERATOR_2_TO_1_PATH,
        discriminator_1_path: Path = settings.DISCRIMINATOR_1_PATH,
        discriminator_2_path: Path = settings.DISCRIMINATOR_2_PATH,
    ):
        self.generator_1_to_2.load_state_dict(torch.load(generator_1_to_2_path))
        self.generator_2_to_1.load_state_dict(torch.load(generator_2_to_1_path))
        self.discriminator_1.load_state_dict(torch.load(discriminator_1_path))
        self.discriminator_2.load_state_dict(torch.load(discriminator_2_path))
