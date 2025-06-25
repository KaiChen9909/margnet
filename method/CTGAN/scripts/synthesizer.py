import numpy as np
import torch
from torch import optim
from torch.nn import functional

from method.CTGAN.scripts.conditional import ConditionalGenerator
from method.CTGAN.scripts.ctgan import Discriminator, Generator
from method.CTGAN.scripts.data_sampler import Sampler


class CTGANSynthesizer(object):
    """Conditional Table GAN Synthesizer.

    This is the core class of the CTGAN project, where the different components
    are orchestrated together.

    For more details about the process, please check the [Modeling Tabular data using
    Conditional GAN](https://arxiv.org/abs/1907.00503) paper.

    Args:
        df:
            Dataframe of data
        domain:
            Dictionary of attribute domains
        embedding_dim (int):
            Size of the random sample passed to the Generator. Defaults to 128.
        gen_dim (tuple or list of ints):
            Size of the output samples for each one of the Residuals. A Residual Layer
            will be created for each one of the values provided. Defaults to (256, 256).
        dis_dim (tuple or list of ints):
            Size of the output samples for each one of the Discriminator Layers. A Linear Layer
            will be created for each one of the values provided. Defaults to (256, 256).
        l2scale (float):
            Wheight Decay for the Adam Optimizer. Defaults to 1e-6.
        batch_size (int):
            Number of data samples to process in each step.
    """

    def __init__(self, domain, embedding_dim=128, gen_dim=(256, 256), dis_dim=(256, 256),
                 l2scale=1e-6, batch_size=500):

        self.domain = domain 

        self.embedding_dim = embedding_dim
        self.gen_dim = gen_dim
        self.dis_dim = dis_dim

        self.l2scale = l2scale
        self.batch_size = batch_size
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.trained_epoches = 0

    def _apply_activate(self, data):
        data_t = []
        st = 0
        for item in self.domain.values():
            if item == 1:
                ed = st + item
                data_t.append(torch.tanh(data[:, st:ed]))
                st = ed
            elif item > 1:
                ed = st + item
                data_t.append(functional.gumbel_softmax(data[:, st:ed], tau=0.2))
                st = ed
            else:
                assert 0

        return torch.cat(data_t, dim=1)

    def _cond_loss(self, data, c, m):
        loss = []
        st = 0
        st_c = 0

        for item in self.domain.values():
            if item == 1:
                st += item

            elif item > 1:
                ed = st + item
                ed_c = st_c + item
                tmp = functional.cross_entropy(
                    data[:, st:ed],
                    torch.argmax(c[:, st_c:ed_c], dim=1),
                    reduction='none'
                )
                loss.append(tmp)
                st = ed
                st_c = ed_c

            else:
                assert 0

        loss = torch.stack(loss, dim=1)

        return (loss * m).sum() / data.size()[0]

    def fit(self, train_data, rho, args=None, discrete_columns=tuple(), batch_size=500):
        """Fit the CTGAN Synthesizer models to the training data.

        Args:
            train_data (pandas.DataFrame):
                Training Data.
            discrete_columns (list-like):
                List of discrete columns to be used to generate the Conditional
                Vector. If ``train_data`` is a Numpy array, this list should
                contain the integer indices of the columns. Otherwise, if it is
                a ``pandas.DataFrame``, this list should contain the column names.
            epochs (int):
                Number of training epochs. Defaults to 300.
        """

        # if not hasattr(self, "transformer"):
        #     self.transformer = DataTransformer()
        #     self.transformer.fit(train_data, discrete_columns)
        # train_data = self.transformer.transform(train_data)
        self._batch_size = batch_size
        self._target_epsilon = args.epsilon

        train_data = train_data.numpy()
        data_sampler = Sampler(train_data, self.domain)

        data_dim = sum(self.domain.values())

        if not hasattr(self, "cond_generator"):
            self.cond_generator = ConditionalGenerator(
                train_data,
                self.domain,
                total_rho=0.1*rho
            )

        if not hasattr(self, "generator"):
            self.generator = Generator(
                self.embedding_dim + self.cond_generator.n_opt,
                self.gen_dim,
                data_dim
            ).to(self.device)

        if not hasattr(self, "discriminator"):
            self.discriminator = Discriminator(
                data_dim + self.cond_generator.n_opt,
                self.dis_dim
            ).to(self.device)

        if not hasattr(self, "optimizerG"):
            self.optimizerG = optim.Adam(
                self.generator.parameters(), lr=2e-4, betas=(0.5, 0.9),
                weight_decay=self.l2scale
            )

        if not hasattr(self, "optimizerD"):
            self.optimizerD = optim.Adam(
                self.discriminator.parameters(), lr=2e-4, betas=(0.5, 0.9))

        # assert self.batch_size % 2 == 0
        mean = torch.zeros(self.batch_size, self.embedding_dim, device=self.device)
        std = mean + 1

        i = 0
        self._G_losses = []
        self._D_losses = []
        self._epsilons = []
        epsilon = 0
        steps = 0
        epoch = 0

        steps_per_epoch = max(len(train_data) // self._batch_size, 1)

        while epsilon < self._target_epsilon:
            for id_ in range(steps_per_epoch):
                fakez = torch.normal(mean=mean, std=std)

                condvec = self.cond_generator.sample(self.batch_size)
                if condvec is None:
                    c1, m1, col, opt = None, None, None, None
                    real = data_sampler.sample(self.batch_size, col, opt)
                else:
                    c1, m1, col, opt = condvec
                    c1 = torch.from_numpy(c1).to(self.device)
                    m1 = torch.from_numpy(m1).to(self.device)
                    fakez = torch.cat([fakez, c1], dim=1)

                    perm = np.arange(self.batch_size)
                    np.random.shuffle(perm)
                    real = data_sampler.sample(self.batch_size, col[perm], opt[perm])
                    c2 = c1[perm]

                fake = self.generator(fakez)
                fakeact = self._apply_activate(fake)

                real = torch.from_numpy(real.astype('float32')).to(self.device)

                if c1 is not None:
                    fake_cat = torch.cat([fakeact, c1], dim=1)
                    real_cat = torch.cat([real, c2], dim=1)
                else:
                    real_cat = real
                    fake_cat = fake

                y_fake = self.discriminator(fake_cat)
                y_real = self.discriminator(real_cat)

                pen = self.discriminator.calc_gradient_penalty(
                    real_cat, fake_cat, self.device)
                loss_d = -(torch.mean(y_real) - torch.mean(y_fake))

                self.optimizerD.zero_grad()
                pen.backward(retain_graph=True)
                loss_d.backward()
                self.optimizerD.step()

                fakez = torch.normal(mean=mean, std=std)
                condvec = self.cond_generator.sample(self.batch_size)

                if condvec is None:
                    c1, m1, col, opt = None, None, None, None
                else:
                    c1, m1, col, opt = condvec
                    c1 = torch.from_numpy(c1).to(self.device)
                    m1 = torch.from_numpy(m1).to(self.device)
                    fakez = torch.cat([fakez, c1], dim=1)

                fake = self.generator(fakez)
                fakeact = self._apply_activate(fake)

                if c1 is not None:
                    y_fake = self.discriminator(torch.cat([fakeact, c1], dim=1))
                else:
                    y_fake = self.discriminator(fakeact)

                if condvec is None:
                    cross_entropy = 0
                else:
                    cross_entropy = self._cond_loss(fake, c1, m1)

                loss_g = -torch.mean(y_fake) + cross_entropy

                self.optimizerG.zero_grad()
                loss_g.backward()
                self.optimizerG.step()

            print("Epoch %d, Loss G: %.4f, Loss D: %.4f" %
                  (self.trained_epoches, loss_g.detach().cpu(), loss_d.detach().cpu()),
                  flush=True)

    def sample(self, n, condition_column=None, condition_value=None):
        """Sample data similar to the training data.

        Args:
            n (int):
                Number of rows to sample.

        Returns:
            numpy.ndarray or pandas.DataFrame
        """

        if condition_column is not None and condition_value is not None:
            condition_info = self.transformer.covert_column_name_value_to_id(
                condition_column, condition_value)
            global_condition_vec = self.cond_generator.generate_cond_from_condition_column_info(
                condition_info, self.batch_size)
        else:
            global_condition_vec = None

        steps = n // self.batch_size + 1
        data = []
        for i in range(steps):
            mean = torch.zeros(self.batch_size, self.embedding_dim)
            std = mean + 1
            fakez = torch.normal(mean=mean, std=std).to(self.device)

            if global_condition_vec is not None:
                condvec = global_condition_vec.copy()
            else:
                condvec = self.cond_generator.sample_zero(self.batch_size)

            if condvec is None:
                pass
            else:
                c1 = condvec
                c1 = torch.from_numpy(c1).to(self.device)
                fakez = torch.cat([fakez, c1], dim=1)

            fake = self.generator(fakez)
            fakeact = self._apply_activate(fake)
            data.append(fakeact.detach().cpu().numpy())

        data = np.concatenate(data, axis=0)
        data = data[:n]

        return self.transformer.inverse_transform(data, None)

    def save(self, path):
        assert hasattr(self, "generator")
        assert hasattr(self, "discriminator")
        assert hasattr(self, "transformer")

        # always save a cpu model.
        device_bak = self.device
        self.device = torch.device("cpu")
        self.generator.to(self.device)
        self.discriminator.to(self.device)

        torch.save(self, path)

        self.device = device_bak
        self.generator.to(self.device)
        self.discriminator.to(self.device)

    @classmethod
    def load(cls, path):
        model = torch.load(path)
        model.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model.generator.to(model.device)
        model.discriminator.to(model.device)
        return model
