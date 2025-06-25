import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
from sklearn.exceptions import ConvergenceWarning
warnings.simplefilter(action='ignore', category=ConvergenceWarning)

import torch
from torch import optim
import matplotlib.pyplot as plt
import numpy as np
from collections import namedtuple

from method.CTGAN.scripts.ctgan import CTGANSynthesizer, Generator, Discriminator
from method.CTGAN.scripts.data_sampler import DataSampler
from method.CTGAN.scripts.transformer import DataTransformer
from method.CTGAN.scripts.rdp_accountant import compute_rdp, get_privacy_spent


SpanInfo = namedtuple("SpanInfo", ["dim", "activation_fn"])



# differential private CTGAN Synthesizer with rdp composition
class DPCTGANSynthesizerRDP(CTGANSynthesizer):
    """
        Basic differential private CTGAN Synthesizer
        Algorithm: https://arxiv.org/pdf/1801.01594.pdf
    Args:
        private (bool):
            Inject random noise during optimization procedure in order to achieve
            differential privacy. Currently only naively inject noise.
            Defaults to ``False``.
        clip_coeff (float):
            Gradient clipping bound. Defaults to ``0.1``.
        sigma (int):
            Noise scale. Defaults to ``2``.
        epsilon (int):
            Differential privacy budget
        delta (float):
            Differential privacy budget

    """
    def __init__(self, embedding_dim=128, generator_dim=(256, 256), discriminator_dim=(256, 256),
                 generator_lr=2e-4, generator_decay=1e-6, discriminator_lr=2e-4, betas=(0.5, 0.99),
                 discriminator_decay=1e-6, batch_size=500, discriminator_steps=1,
                 log_frequency=True, verbose=False, epochs=300, pac=10, device='cpu',
                 clip_coeff=0.1, sigma=1, target_epsilon=3, target_delta=1e-5, rho_total = None, rho_used = None, domain = None):

        assert batch_size % 2 == 0 , f"Batch size: {batch_size}"

        super(DPCTGANSynthesizerRDP, self).__init__(embedding_dim, generator_dim, discriminator_dim,
                         generator_lr, generator_decay, discriminator_lr, betas,
                         discriminator_decay, batch_size, discriminator_steps,
                         log_frequency, verbose, epochs, pac, device)

        self._clip_coeff = clip_coeff
        self._sigma = sigma
        self._target_epsilon = target_epsilon
        self._target_delta = target_delta
        self._rho = rho_total
        self._rho_used = rho_used 
        self.domain = domain

        print(f'Init CTGAN with differential privacy')


    def get_config(self):
        return f"Clip Coefficient: {self._clip_coeff}\n" \
               f"Sigma: {self._sigma}\n" \
               f"Target Epsilon: {self._target_epsilon}\n" \
               f"Target delta: {self._target_delta}\n"


    def fit(self, train_data, discrete_columns=tuple()):
        self._validate_discrete_columns(train_data, discrete_columns)

        self._transformer = DataTransformer()
        self._transformer.fit(train_data, discrete_columns)
        train_data = self._transformer.transform(train_data)

        if len(discrete_columns) > 0:
            rho_sampler = 0.1*(self._rho-self._rho_used)
        else:
            rho_sampler = 0.0

        self._data_sampler = DataSampler(
            train_data,
            self._transformer.output_info_list,
            self._log_frequency,
            rho_sampler
        )

        data_dim = self._transformer.output_dimensions

        self._generator = Generator(
            self._embedding_dim + self._data_sampler.dim_cond_vec(),
            self._generator_dim,
            data_dim
        ).to(self._device)

        discriminator = Discriminator(
            data_dim + self._data_sampler.dim_cond_vec(),
            self._discriminator_dim,
            pac=self.pac
        ).to(self._device)

        optimizerG = optim.Adam(
            self._generator.parameters(), lr=self._generator_lr, betas=self._betas,
            weight_decay=self._generator_decay
        )

        optimizerD = optim.Adam(
            discriminator.parameters(), lr=self._discriminator_lr,
            betas=self._betas, weight_decay=self._discriminator_decay
        )

        mean = torch.zeros(self._batch_size, self._embedding_dim, device=self._device)
        std = mean + 1

        i = 0
        self._G_losses = []
        self._D_losses = []
        self._epsilons = []
        epsilon = 0
        steps = 0
        epoch = 0

        steps_per_epoch = max(len(train_data) // self._batch_size, 1)

        max_lmbd = 4095
        lmbds = np.arange(2, max_lmbd + 1)
        rdp_plus = lmbds*(self._rho_used + rho_sampler)

        while epsilon < self._target_epsilon:

            for id_ in range(steps_per_epoch):

                ############################
                # (1) Update D network
                ###########################
                for n in range(self._discriminator_steps):

                    for name, param in discriminator.named_parameters():
                        if param.grad is not None:
                            # clip gradient by the threshold C
                            clipped_gradient = param.grad / max(1, torch.norm(param.grad,
                                                                              2) / self._clip_coeff)
                            # generate random noise from a Gaussian distribution
                            noise = torch.DoubleTensor(param.size()) \
                                .normal_(0, (self._sigma * self._clip_coeff) ** 2) \
                                .to(self._device)

                            param.grad = (clipped_gradient + noise).float()
                    steps += 1

                    # train with fake
                    fakez = torch.normal(mean=mean, std=std)
                    condvec = self._data_sampler.sample_condvec(self._batch_size)

                    if condvec is None:
                        c1, m1, col, opt = None, None, None, None
                        real = self._data_sampler.sample_data(self._batch_size, col, opt)
                    else:
                        c1, m1, col, opt = condvec
                        c1 = torch.from_numpy(c1).to(self._device)
                        m1 = torch.from_numpy(m1).to(self._device)
                        fakez = torch.cat([fakez, c1], dim=1)

                        perm = np.arange(self._batch_size)
                        np.random.shuffle(perm)
                        real = self._data_sampler.sample_data(
                            self._batch_size, col[perm], opt[perm])
                        c2 = c1[perm]

                    fake = self._generator(fakez)
                    fakeact = self._apply_activate(fake)

                    real = torch.from_numpy(real.astype('float32')).to(self._device)

                    if c1 is not None:
                        fake_cat = torch.cat([fakeact, c1], dim=1)
                        real_cat = torch.cat([real, c2], dim=1)
                    else:
                        real_cat = real
                        fake_cat = fakeact

                    y_fake = discriminator(fake_cat)
                    y_real = discriminator(real_cat)

                    pen = discriminator.calc_gradient_penalty(
                        real_cat, fake_cat, self._device, self.pac)

                    loss_d = -(torch.mean(y_real) - torch.mean(y_fake))  # + pen

                    optimizerD.zero_grad()
                    pen.backward(retain_graph=True)
                    loss_d.backward()  # Calculate gradients for D in backward pass
                    optimizerD.step()  # Update D

                ############################
                # (2) Update G network
                ###########################
                fakez = torch.normal(mean=mean, std=std)
                condvec = self._data_sampler.sample_condvec(self._batch_size)

                if condvec is None:
                    c1, m1, col, opt = None, None, None, None
                else:
                    c1, m1, col, opt = condvec
                    c1 = torch.from_numpy(c1).to(self._device)
                    m1 = torch.from_numpy(m1).to(self._device)
                    fakez = torch.cat([fakez, c1], dim=1)

                fake = self._generator(fakez)  # Generate fake data batch with G
                fakeact = self._apply_activate(fake)

                # Since we just updated D, perform another forward pass
                if c1 is not None:
                    y_fake = discriminator(torch.cat([fakeact, c1], dim=1))
                else:
                    y_fake = discriminator(fakeact)

                if condvec is None:
                    cross_entropy = 0
                else:
                    cross_entropy = self._cond_loss(fake, c1, m1)

                # Calculate G's loss based on this output
                loss_g = -torch.mean(y_fake) + cross_entropy

                optimizerG.zero_grad()
                loss_g.backward()  # Calculate gradients for G
                optimizerG.step()  # Update G

            # Save losses for plotting later
            self._G_losses.append(loss_g.item())
            self._D_losses.append(loss_d.item())
            epoch += 1

            # calculate current privacy cost using the accountant
            max_lmbd = 4095
            lmbds = range(2, max_lmbd + 1)
            rdp = compute_rdp(self._batch_size / len(train_data),
                              self._sigma, steps, lmbds)
            rdp += rdp_plus
            epsilon, _, _ = get_privacy_spent(lmbds, rdp, None, self._target_delta)
            self._epsilons.append(epsilon)

            # Output training stats
            if self._verbose:
                print(f"Epoch {i + 1}, "
                      f"Loss G: {loss_g.detach().cpu(): .4f}, "
                      f"Loss D: {loss_d.detach().cpu(): .4f}, "
                      f"Epsilon: {epsilon:.4f}", flush=True)
                i += 1

    def plot_losses(self, save=False):
        plt.figure(figsize=(10, 5))
        plt.title("Generator and Discriminator Loss during training")
        plt.plot(self._G_losses, label='G')
        plt.plot(self._D_losses, label='D')
        plt.xlabel('iterations')
        plt.ylabel('Loss')
        intervals = len(self._G_losses) // 5 if len(self._G_losses) > 5 else len(self._G_losses)
        x_ticks = np.arange(0, len(self._G_losses), intervals)
        plt.xticks(x_ticks)
        plt.legend()
        if save:
            plt.savefig('losses.png')
        plt.show()

    def sample(self, num_samples, preprocesser=None, parent_dir=None):
        """Sample data similar to the training data.

        Choosing a condition_column and condition_value will increase the probability of the
        discrete condition_value happening in the condition_column.
        Args:
            n (int):
                Number of rows to sample.
            condition_column (string):
                Name of a discrete column.
            condition_value (string):
                Name of the category in the condition_column which we wish to increase the
                probability of happening.
        Returns:
            numpy.ndarray or pandas.DataFrame
        """
        # if condition_column is not None and condition_value is not None:
        #     condition_info = self._transformer.convert_column_name_value_to_id(
        #         condition_column, condition_value)
        #     global_condition_vec = self._data_sampler.generate_cond_from_condition_column_info(
        #         condition_info, self._batch_size)
        # else:
        #     global_condition_vec = None
        global_condition_vec = None

        steps = num_samples // self._batch_size + 1
        data = []
        for i in range(steps):
            mean = torch.zeros(self._batch_size, self._embedding_dim)
            std = mean + 1
            fakez = torch.normal(mean=mean, std=std).to(self._device)

            if global_condition_vec is not None:
                condvec = global_condition_vec.copy()
            else:
                condvec = self._data_sampler.sample_original_condvec(self._batch_size)

            if condvec is None:
                pass
            else:
                c1 = condvec
                c1 = torch.from_numpy(c1).to(self._device)
                fakez = torch.cat([fakez, c1], dim=1)

            fake = self._generator(fakez)
            fakeact = self._apply_activate(fake)
            data.append(fakeact.detach().cpu().numpy())

        data = np.concatenate(data, axis=0)
        data = data[:num_samples]
        data = self._transformer.inverse_transform(data)

        if preprocesser is not None and parent_dir is not None:
            preprocesser.reverse_data(data, parent_dir)

        return data

    def save(self, path):
        assert hasattr(self, "generator")
        assert hasattr(self, "discriminator")

        # always save a cpu model.
        device_bak = self.device
        self.device = torch.device("cpu")
        self.generator.to(self.device)
        self.discriminator.to(self.device)

        torch.save(self, path)

        self.device = device_bak
        self.generator.to(self.device)
        self.discriminator.to(self.device)