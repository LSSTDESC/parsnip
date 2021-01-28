from tqdm import tqdm
import functools
import multiprocessing
import numpy as np
import os
import sys

from astropy.table import Table, vstack
import extinction
import sncosmo

import torch
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F

from .astronomical_object import ParsnipObject


class ParsnipDataLoader():
    def __init__(self, dataset, autoencoder, shuffle=False):
        self.dataset = dataset
        self.autoencoder = autoencoder
        self.shuffle = shuffle

        if self.shuffle:
            self._ordered_objects = np.random.permutation(self.dataset.objects)
        else:
            self._ordered_objects = self.dataset.objects

    def __len__(self):
        return len(self.dataset.objects)

    def __getitem__(self, index):
        objects = self._ordered_objects[index]
        return self.autoencoder.get_data(objects)

    def __iter__(self):
        """Setup iteration over batches"""
        self.batch_idx = 0
        return self

    def __next__(self):
        """Retrieve the next batch"""
        start = self.batch_idx * self.autoencoder.batch_size
        end = (self.batch_idx + 1) * self.autoencoder.batch_size
        if start >= len(self):
            raise StopIteration
        else:
            self.batch_idx += 1
            return self[start:end]

    @property
    def num_batches(self):
        """Return the number of batches that this dataset is split into"""
        return 1 + (len(self) + 1) // self.autoencoder.batch_size


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dilation, batchnorm=False,
                 concat=True, maxpool_input=True):
        super().__init__()

        self.concat = concat

        self.conv = nn.Conv1d(in_channels, out_channels, 5, dilation=dilation,
                              padding=2*dilation)

        if batchnorm:
            self.batchnorm = nn.BatchNorm1d(out_channels)
        else:
            self.batchnorm = None

        if maxpool_input:
            # Apply a max pool to the input to keep track of where we had observations.
            self.maxpool_input = nn.MaxPool1d(5, 1, dilation=dilation)

            # Padding is broken for dilated max pools in pytorch 1.7.0. Do it manually.
            self.maxpool_input_pad = nn.ConstantPad1d(2*dilation, 0.)
        else:
            self.maxpool_input = None

    def forward(self, pool_input, x):
        out = self.conv(x)

        if self.maxpool_input is not None:
            pool_input = self.maxpool_input(self.maxpool_input_pad(pool_input))

        if self.batchnorm is not None:
            out = self.batchnorm(out)

        out = F.relu(out)
        if self.concat:
            out = torch.cat([pool_input, out], 1)

        return pool_input, out


class LightCurveAutoencoder(nn.Module):
    def __init__(self, name, bands, device='cpu', batch_size=128, latent_size=3,
                 min_wave=1000., max_wave=11000., spectrum_bins=300, max_redshift=4.,
                 band_oversampling=51, time_window=300, time_pad=100, time_sigma=20.,
                 magsys='ab', error_floor=0.01, learning_rate=1e-3,
                 min_learning_rate=1e-5, penalty=1e-3, correct_mw_extinction=False,
                 correct_background=False, augment=True, augment_mask_fraction=0.,
                 maxpool_input=True):
        super().__init__()

        # Figure out if CUDA is available and if we want to use it.
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.batch_size = batch_size

        self.name = name
        self.latent_size = latent_size
        self.min_wave = min_wave
        self.max_wave = max_wave
        self.spectrum_bins = spectrum_bins
        self.max_redshift = max_redshift
        self.band_oversampling = band_oversampling
        self.time_window = time_window
        self.time_pad = time_pad
        self.time_sigma = time_sigma
        self.magsys = magsys
        self.error_floor = error_floor
        self.learning_rate = learning_rate
        self.min_learning_rate = min_learning_rate
        self.penalty = penalty
        self.correct_mw_extinction = correct_mw_extinction
        self.correct_background = correct_background
        self.augment = augment
        self.augment_mask_fraction = augment_mask_fraction
        self.maxpool_input = maxpool_input

        # Setup the bands
        self.bands = bands
        self.band_map = {j: i for i, j in enumerate(bands)}
        self._setup_band_weights()
        self._calculate_band_wave_effs()

        # Setup the color law
        color_law = extinction.fm07(self.model_wave, 1.)
        self.color_law = torch.FloatTensor(color_law).to(self.device)

        # Setup the timing
        self.center_time_bin = self.time_window // 2
        self.input_times = (torch.arange(self.time_window, device=self.device)
                            - self.center_time_bin)

        # Build the model
        self.build_model()

        # Setup the training
        self.epoch = 0
        self.optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, factor=0.5, verbose=True
        )

        # Send the model to the desired device
        self.to(self.device)

    def to(self, device):
        """Send the model to the given device"""
        new_device = torch.device(device if torch.cuda.is_available() else 'cpu')

        if self.device == new_device:
            # Already on that device
            return

        self.device = new_device

        # Send all of the weights
        super().to(self.device)

        # Send all of the variables that we create manually
        self.color_law = self.color_law.to(self.device)
        self.input_times = self.input_times.to(self.device)

        self.band_interpolate_locations = \
            self.band_interpolate_locations.to(self.device)
        self.band_interpolate_weights = self.band_interpolate_weights.to(self.device)

    def _setup_band_weights(self):
        """Setup the interpolation for the band weights used for photometry"""
        # Build the model in log wavelength
        model_log_wave = np.linspace(np.log10(self.min_wave), np.log10(self.max_wave),
                                     self.spectrum_bins)
        model_spacing = model_log_wave[1] - model_log_wave[0]

        band_spacing = model_spacing / self.band_oversampling
        band_max_log_wave = (
            np.log10(self.max_wave * (1 + self.max_redshift))
            + band_spacing
        )

        # Oversampling must be odd.
        assert self.band_oversampling % 2 == 1
        pad = (self.band_oversampling - 1) // 2
        band_log_wave = np.arange(np.log10(self.min_wave), band_max_log_wave,
                                  band_spacing)
        band_wave = 10**(band_log_wave)
        band_pad_log_wave = np.arange(
            np.log10(self.min_wave) - band_spacing * pad,
            band_max_log_wave + band_spacing * pad,
            band_spacing
        )
        band_pad_dwave = (
            10**(band_pad_log_wave + band_spacing / 2.)
            - 10**(band_pad_log_wave - band_spacing / 2.)
        )

        ref = sncosmo.get_magsystem(self.magsys)

        band_weights = []

        for band_name in self.bands:
            band = sncosmo.get_bandpass(band_name)
            band_transmission = band(10**(band_pad_log_wave))

            # Convolve the bands to match the sampling of the spectrum.
            band_conv_transmission = np.convolve(
                band_transmission * band_pad_dwave,
                np.ones(self.band_oversampling),
                mode='valid'
            )

            band_weight = (
                band_wave
                * band_conv_transmission
                / sncosmo.constants.HC_ERG_AA
                / ref.zpbandflux(band)
                * 10**(0.4 * -20.)
            )

            band_weights.append(band_weight)

        # Get the locations that should be sampled at redshift 0. We can scale these to
        # get the locations at any redshift.
        band_interpolate_locations = torch.arange(
            0,
            self.spectrum_bins * self.band_oversampling,
            self.band_oversampling
        )

        # Save the variables that we need to do interpolation.
        self.band_interpolate_locations = band_interpolate_locations.to(self.device)
        self.band_interpolate_spacing = band_spacing
        self.band_interpolate_weights = torch.FloatTensor(band_weights).to(self.device)
        self.model_wave = 10**(model_log_wave)

    def _calculate_band_wave_effs(self):
        """Calculate the effective wavelength of each band"""
        band_wave_effs = []
        for band_name in self.bands:
            band = sncosmo.get_bandpass(band_name)
            band_wave_effs.append(band.wave_eff)

        self.band_wave_effs = np.array(band_wave_effs)

    def calculate_band_weights(self, redshifts):
        """Calculate the band weights for a given set of redshifts

        We have precomputed the weights for each bandpass, so we simply interpolate
        those weights at the desired redshifts. We are working in log-wavelength, so a
        change in redshift just gives us a shift in indices.
        """
        # Figure out the locations to sample at for each redshift.
        locs = (
            self.band_interpolate_locations
            + torch.log10(1 + redshifts)[:, None] / self.band_interpolate_spacing
        )
        flat_locs = locs.flatten()

        # Linear interpolation
        int_locs = flat_locs.long()
        remainders = flat_locs - int_locs

        start = self.band_interpolate_weights[..., int_locs]
        end = self.band_interpolate_weights[..., int_locs + 1]

        flat_result = remainders * end + (1 - remainders) * start
        result = flat_result.reshape((-1,) + locs.shape).permute(1, 2, 0)

        # We need an extra term of 1 + z from the filter contraction.
        result /= (1 + redshifts)[:, None, None]

        return result

    def test_band_weights(self, redshift, source='salt2-extended'):
        """Test the band weights by comparing sncosmo photometry to the photometry
        calculated by this class.
        """
        model = sncosmo.Model(source=source)

        # sncosmo photometry
        model.set(z=redshift)
        sncosmo_photometry = model.bandflux(self.bands, 0., zp=-20., zpsys=self.magsys)

        # autoencoder photometry
        model.set(z=0.)
        model_flux = model._flux(0., self.model_wave)[0]
        band_weights = self.calculate_band_weights(
            torch.FloatTensor([redshift]))[0].numpy()
        autoencoder_photometry = np.sum(model_flux[:, None] * band_weights, axis=0)

        print(f"z = {redshift}")
        print(f"sncosmo photometry:     {sncosmo_photometry}")
        print(f"autoencoder photometry: {autoencoder_photometry}")
        print(f"ratio:                  {autoencoder_photometry / sncosmo_photometry}")

    def preprocess(self, dataset, threads=16, chunksize=64):
        """Preprocess a dataset"""
        if threads == 1:
            # Run on a single core without multiprocessing
            for obj in tqdm(dataset.objects, file=sys.stdout):
                obj.preprocess(self)
        else:
            # Run with multiprocessing in multiple threads.
            # For multiprocessing, everything has to be on the CPU. Send the current
            # model to the CPU, and keep track of where it was.
            # TODO: all that we need is the settings. Refactor the code so that we
            # don't have to move the whole model around.
            current_device = self.device
            self.to('cpu')

            func = functools.partial(ParsnipObject.preprocess, autoencoder=self)

            with multiprocessing.Pool(threads) as p:
                data = list(tqdm(p.imap(func, dataset.objects, chunksize=chunksize),
                                 total=len(dataset.objects),
                                 file=sys.stdout))

            # The outputs don't get saved since the objects are in a different thread.
            # Save them manually.
            for obj, obj_data in zip(dataset.objects, data):
                obj.preprocess_data = obj_data

            # Move us back to the appropriate device
            self.to(current_device)

    def get_data(self, objects):
        """Extract the data needed from an object or set of objects needed to run the
        autoencoder.
        """
        # Check if we have a list of objects or a single one and handle it
        # appropriately.
        try:
            iter(objects)
            single = False
        except TypeError:
            single = True
            objects = [objects]

        input_data = []
        compare_data = []
        redshifts = []

        if self.augment:
            time_shifts = np.round(
                np.random.normal(0., self.time_sigma, len(objects))).astype(int)
            amp_scales = np.exp(np.random.normal(0, 0.5, len(objects)))
        else:
            time_shifts = np.zeros(len(objects), dtype=int)
            amp_scales = np.ones(len(objects))

        # Extract the data from each object. Numpy is much faster than torch for
        # vector operations currently, so do as much as we can in numpy before
        # converting to torch tensors.

        times = []
        fluxes = []
        flux_errors = []
        band_indices = []
        time_indices = []
        batch_indices = []
        compare_data = []
        compare_band_indices = []
        redshifts = []

        for idx, obj in enumerate(objects):
            data = obj.preprocess_data

            # Shift and scale the grid data
            obj_times = data['time'] + time_shifts[idx]
            obj_flux = data['flux'] * amp_scales[idx]
            obj_flux_error = data['flux_error'] * amp_scales[idx]
            obj_band_indices = data['band_indices']
            obj_time_indices = data['time_indices'] + time_shifts[idx]
            obj_batch_indices = np.ones_like(obj_band_indices) * idx

            # Compute the weight that will be used for comparing the model to the input
            # data. We use the observed error with an error floor.
            obj_compare_weight = 1. / (obj_flux_error**2 + self.error_floor**2)

            # Stack all of the data that will be used for comparisons and convert it to
            # torch tensor.
            obj_compare_data = torch.FloatTensor(np.vstack([
                obj_times,
                obj_flux,
                obj_flux_error,
                obj_compare_weight,
            ]))
            compare_band_indices.append(torch.LongTensor(obj_band_indices))

            times.append(obj_times)
            fluxes.append(obj_flux)
            flux_errors.append(obj_flux_error)

            band_indices.append(obj_band_indices)
            time_indices.append(obj_time_indices)
            batch_indices.append(obj_batch_indices)
            compare_data.append(obj_compare_data.T)

            redshifts.append(obj.metadata['redshift'])

        # Gather the input data.
        fluxes = np.concatenate(fluxes)
        flux_errors = np.concatenate(flux_errors)
        band_indices = np.concatenate(band_indices)
        time_indices = np.concatenate(time_indices)
        batch_indices = np.concatenate(batch_indices)
        redshifts = np.array(redshifts)

        # Throw out inputs that are outside of our window.
        mask = (time_indices >= 0) & (time_indices < self.time_window)

        if self.augment and self.augment_mask_fraction > 0.:
            # Throw out some of the observations in the input. We keep them in the
            # output for comparisons. This is effectively a version of dropout.
            mask &= np.random.rand(len(fluxes)) < self.augment_mask_fraction

        fluxes = fluxes[mask]
        flux_errors = flux_errors[mask]
        band_indices = band_indices[mask]
        time_indices = time_indices[mask]
        batch_indices = batch_indices[mask]

        # Add the error floor to the flux errors.
        add_flux_errors = self.error_floor

        if self.augment:
            # If we are augmenting, add the error floor to the fluxes
            fluxes += np.random.normal(0, add_flux_errors, fluxes.shape)

        # Use the logarithm of the error as a weight. We scale things so that 1
        # represents a measurement with an error of 1, 2 represents a measurement with
        # an error of 0.1, etc. We set the error for unobserved points to be 0
        # (equivalent to an error of 10 which would be extremely large).
        weights = 1 + -0.5 * np.log10(flux_errors**2 + add_flux_errors**2)

        # Build a grid for the input
        grid_flux = np.zeros((len(objects), len(self.bands), self.time_window))
        grid_weights = np.zeros_like(grid_flux)

        grid_flux[batch_indices, band_indices, time_indices] = fluxes
        grid_weights[batch_indices, band_indices, time_indices] = weights

        # Stack the input data
        input_data = np.concatenate([
            grid_flux,
            grid_weights,
            redshifts[:, None, None].repeat(self.time_window, axis=2)
        ], axis=1)

        # Convert to torch tensors
        input_data = torch.FloatTensor(input_data)
        redshifts = torch.FloatTensor(redshifts)

        # Pad all of the compare data to have the same shape.
        compare_data = nn.utils.rnn.pad_sequence(compare_data, batch_first=True)
        compare_data = compare_data.permute(0, 2, 1)
        compare_band_indices = nn.utils.rnn.pad_sequence(compare_band_indices,
                                                         batch_first=True)

        if single:
            return (input_data[0], compare_data[0], redshifts[0],
                    compare_band_indices[0])
        else:
            return input_data, compare_data, redshifts, compare_band_indices

    def build_model(self):
        """Build the model"""
        input_size = len(self.bands) * 2 + 1

        self.conv_encodes = nn.ModuleList([
            ConvBlock(input_size, 20, 1, maxpool_input=self.maxpool_input),
            ConvBlock(input_size + 20, 20, 1, maxpool_input=self.maxpool_input),
            ConvBlock(input_size + 20, 40, 2, maxpool_input=self.maxpool_input),
            ConvBlock(input_size + 40, 40, 4, maxpool_input=self.maxpool_input),
            ConvBlock(input_size + 40, 60, 8, maxpool_input=self.maxpool_input),
            ConvBlock(input_size + 60, 80, 16, maxpool_input=self.maxpool_input),
            ConvBlock(input_size + 80, 100, 32, maxpool_input=self.maxpool_input),
            ConvBlock(input_size + 100, 120, 64, maxpool_input=self.maxpool_input,
                      concat=False),
        ])

        self.encode_1 = nn.Conv1d(120, 40, 1)
        self.encode_2 = nn.Conv1d(40, 40, 1)

        self.conv_time = nn.Conv1d(40, 1, 1)
        self.softmax_time = nn.Softmax(dim=1)

        self.fc_encoding_mu = nn.Linear(40, self.latent_size + 1)
        self.fc_encoding_logvar = nn.Linear(40, self.latent_size + 2)

        self.decode_1 = nn.Conv1d(self.latent_size + 1, 40, 1)
        self.decode_2 = nn.Conv1d(40, 40, 1)
        self.decode_3 = nn.Conv1d(40, self.spectrum_bins, 1)

    def encode(self, input_data):
        e = input_data
        pool_input = input_data
        for conv_encode in self.conv_encodes:
            pool_input, e = conv_encode(pool_input, e)

        e1 = F.relu(self.encode_1(e))
        e2 = self.encode_2(e1)

        # Reference time: we calculate its mean with a special layer that is invariant
        # to translations of the input.
        t_vec = torch.nn.functional.softmax(torch.squeeze(self.conv_time(e2), 1), dim=1)
        ref_time_mu = torch.sum(t_vec * self.input_times, 1) / self.time_sigma

        # Rest of the encoding.
        ee, max_inds = torch.max(e2, 2)
        encoding_mu = self.fc_encoding_mu(ee)
        encoding_logvar = self.fc_encoding_logvar(ee)

        # Prepend the time mu value to get the full encoding.
        encoding_mu = torch.cat([ref_time_mu[:, None], encoding_mu], 1)

        # Constrain the logvar so that it doesn't go to crazy values and throw
        # everything off with floating point precision errors. This will not be a
        # concern for a properly trained model, but things can go wrong early in the
        # training at high learning rates.
        encoding_logvar = torch.clamp(encoding_logvar, None, 5.)

        return encoding_mu, encoding_logvar

    def decode(self, encoding, ref_times, color, times, redshifts, band_indices,
               amplitude=None):
        repeat_encoding = encoding[:, :, None].expand((-1, -1, times.shape[1]))
        use_times = (
            (times - ref_times[:, None])
            / (self.time_window // 2)
            / (1 + redshifts[:, None])
        )

        stack_encoding = torch.cat([repeat_encoding, use_times[:, None, :]], 1)

        # Decode
        d1 = torch.tanh(self.decode_1(stack_encoding))
        d2 = torch.tanh(self.decode_2(d1))
        model_spectra = F.softplus(self.decode_3(d2))

        # Apply colors
        apply_colors = 10**(-0.4 * color[:, None] * self.color_law[None, :])
        model_spectra = model_spectra * apply_colors[..., None]

        # Figure out the weights for each band
        band_weights = self.calculate_band_weights(redshifts)
        num_batches = band_indices.shape[0]
        num_observations = band_indices.shape[1]
        batch_indices = (
            torch.arange(num_batches, device=encoding.device)
            .repeat_interleave(num_observations)
        )
        obs_band_weights = (
            band_weights[batch_indices, :, band_indices.flatten()]
            .reshape((num_batches, num_observations, -1))
            .permute(0, 2, 1)
        )

        # Sum over each filter.
        model_flux = torch.sum(model_spectra * obs_band_weights, axis=1)

        if amplitude is not None:
            model_flux = model_flux * amplitude[:, None]
            model_spectra = model_spectra * amplitude[:, None, None]

        return model_spectra, model_flux

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def sample(self, encoding_mu, encoding_logvar):
        sample_encoding = self.reparameterize(encoding_mu, encoding_logvar)

        ref_times = sample_encoding[:, 0] * self.time_sigma
        color = sample_encoding[:, 1]
        encoding = sample_encoding[:, 2:]

        # Constrain the color and reference time so that things don't go to crazy values
        # and throw everything off with floating point precision errors. This will not
        # be a concern for a properly trained model, but things can go wrong early in
        # the training at high learning rates.
        color = torch.clamp(color, -10., 10.)
        ref_times = torch.clamp(ref_times, -10. * self.time_sigma,
                                10. * self.time_sigma)

        return ref_times, color, encoding

    def forward(self, input_data, compare_data, redshifts, band_indices):
        encoding_mu, encoding_logvar = self.encode(input_data)

        ref_times, color, encoding = self.sample(encoding_mu, encoding_logvar)

        model_spectra, model_flux = self.decode(
            encoding, ref_times, color, compare_data[:, 0], redshifts, band_indices
        )

        flux = compare_data[:, 1]
        weight = compare_data[:, 3]

        # Analytically evaluate the conditional distribution for the amplitude.
        amplitude_mu, amplitude_logvar = self.compute_amplitude(weight, model_flux,
                                                                flux)
        amplitude = self.reparameterize(amplitude_mu, amplitude_logvar)
        model_flux = model_flux * amplitude[:, None]
        model_spectra = model_spectra * amplitude[:, None, None]

        return (encoding_mu, encoding_logvar, amplitude_mu, amplitude_logvar, ref_times,
                color, encoding, amplitude, model_flux, model_spectra)

    def compute_amplitude(self, weight, model_flux, flux):
        num = torch.sum(weight * model_flux * flux, axis=1)
        denom = torch.sum(weight * model_flux * model_flux, axis=1)

        amplitude_mu = num / denom
        amplitude_logvar = torch.log(1. / denom)

        return amplitude_mu, amplitude_logvar

    def loss_function(self, compare_data, encoding_mu, encoding_logvar, amplitude_mu,
                      amplitude_logvar, amplitude, model_flux, model_spectra):
        flux = compare_data[:, 1]
        weight = compare_data[:, 3]

        # Reconstruction likelihood
        nll = torch.sum(0.5 * weight * (flux - model_flux)**2)

        # Regularization of spectra
        diff = (
            (model_spectra[:, 1:, :] - model_spectra[:, :-1, :])
            / (model_spectra[:, 1:, :] + model_spectra[:, :-1, :])
        )
        penalty = self.penalty * torch.sum(diff**2)

        # Amplitude probability
        amp_prob = -0.5 * torch.sum((amplitude - amplitude_mu)**2 /
                                    amplitude_logvar.exp())

        # KL divergence
        kld = -0.5 * torch.sum(1 + encoding_logvar - encoding_mu.pow(2) -
                               encoding_logvar.exp())

        return nll + penalty + kld + amp_prob

    def fit(self, dataset, max_epochs=1000, augments=1):
        while self.epoch < max_epochs:
            loader = ParsnipDataLoader(dataset, self, shuffle=True)
            self.train()
            train_loss = 0
            train_count = 0

            with tqdm(range(loader.num_batches * augments), file=sys.stdout) as pbar:
                for augment in range(augments):
                    for batch_data in loader:
                        input_data, compare_data, redshifts, band_indices = batch_data

                        input_data = input_data.to(self.device)
                        compare_data = compare_data.to(self.device)
                        redshifts = redshifts.to(self.device)
                        band_indices = band_indices.to(self.device)

                        self.optimizer.zero_grad()
                        encoding_mu, encoding_logvar, amplitude_mu, amplitude_logvar, \
                            ref_times, color, encoding, amplitude, model_flux, \
                            model_spectra = \
                            self(input_data, compare_data, redshifts, band_indices)

                        loss = self.loss_function(compare_data, encoding_mu,
                                                  encoding_logvar, amplitude_mu,
                                                  amplitude_logvar, amplitude,
                                                  model_flux, model_spectra)

                        loss.backward()
                        train_loss += loss.item()
                        self.optimizer.step()

                        train_count += len(input_data)

                        pbar.set_description(
                            f'Epoch {self.epoch}: Loss: {train_loss / train_count:.4f} '
                            f'({loss.item() / len(input_data):.4f})',
                            refresh=False
                        )
                        pbar.update()

            pbar.refresh()
            self.scheduler.step(train_loss)
            os.makedirs('./models/', exist_ok=True)
            torch.save(self.state_dict(), f'./models/{self.name}.pt')

            # Check if the learning rate is below our threshold, and exit if it is.
            lr = self.optimizer.param_groups[0]['lr']
            if lr < self.min_learning_rate:
                break

            self.epoch += 1

    def load(self):
        """Load the model weights"""
        self.load_state_dict(torch.load(f'./models/{self.name}.pt', self.device))

    def _predict(self, input_data, compare_data, redshifts, band_indices):
        # Move everything to the right device
        input_data = input_data.to(self.device)
        compare_data = compare_data.to(self.device)
        redshifts = redshifts.to(self.device)
        band_indices = band_indices.to(self.device)

        result = self.forward(input_data, compare_data, redshifts, band_indices)
        result = [i.cpu().detach().numpy() for i in result]

        data = {
            'input_data': input_data.cpu().detach().numpy(),
            'compare_data': compare_data.cpu().detach().numpy(),
            'redshifts': redshifts.cpu().detach().numpy(),
            'band_indices': band_indices.cpu().detach().numpy(),
        }

        # Extract result keys
        keys = [
            'encoding_mu',
            'encoding_logvar',
            'amplitude_mu',
            'amplitude_logvar',
            'ref_times',
            'color',
            'encoding',
            'amplitude',
            'model_flux',
            'model_spectra',
        ]
        for key, val in zip(keys, result):
            data[key] = val

        # Get rid of extra dimensions
        # This is slow and ragged arrays are terrible... is there a better way to do
        # it? Not a major issue since this isn't called very often...
        batch_size = len(input_data)
        compare_data = np.zeros(batch_size, dtype=object)
        band_indices = np.zeros(batch_size, dtype=object)
        model_flux = np.zeros(batch_size, dtype=object)
        model_spectra = np.zeros(batch_size, dtype=object)
        for idx in range(len(input_data)):
            mask = data['compare_data'][idx, 3] != 0.
            compare_data[idx] = data['compare_data'][idx, :, mask]
            band_indices[idx] = data['band_indices'][idx, mask]
            model_flux[idx] = data['model_flux'][idx, mask]
            model_spectra[idx] = data['model_spectra'][idx, :, mask]
        data['compare_data'] = compare_data
        data['band_indices'] = band_indices
        data['model_flux'] = model_flux
        data['model_spectra'] = model_spectra

        # Turn everything into an astropy table
        tab = Table(data)

        return tab

    def predict_dataset(self, dataset):
        result = []

        loader = ParsnipDataLoader(dataset, self)

        for batch_data in loader:
            result.append(self._predict(*batch_data))

        result = vstack(result)

        return result

    def _predict_single(self, obj, pred_times, pred_bands, count):
        data = self.get_data(obj)
        input_data, compare_data, redshifts, band_indices = data

        # Add batch dimension
        input_data = input_data[None, :, :].to(self.device)
        compare_data = compare_data[None, :, :].to(self.device)
        redshifts = redshifts.reshape(1).to(self.device)
        band_indices = band_indices[None, :].to(self.device)

        pred_times = torch.FloatTensor(pred_times)[None, :].to(self.device)
        pred_bands = torch.LongTensor(pred_bands)[None, :].to(self.device)

        if count > 0:
            # Predict multiple light curves
            input_data = input_data.repeat(count, 1, 1)
            compare_data = compare_data.repeat(count, 1, 1)
            redshifts = redshifts.repeat(count)
            band_indices = band_indices.repeat(count, 1)

            pred_times = pred_times.repeat(count, 1)
            pred_bands = pred_bands.repeat(count, 1)

        # Sample VAE parameters
        encoding_mu, encoding_logvar, amplitude_mu, amplitude_logvar, ref_times, \
            color, encoding, amplitude, model_flux, model_spectra = self.forward(
                input_data, compare_data, redshifts, band_indices)

        # Do the predictions
        model_spectra, model_flux = self.decode(
            encoding,
            ref_times,
            color,
            pred_times,
            redshifts,
            pred_bands,
            amplitude,
        )

        model_flux = model_flux.cpu().detach().numpy()
        model_spectra = model_spectra.cpu().detach().numpy()

        if count == 0:
            # Get rid of the batch index
            model_flux = model_flux[0]
            model_spectra = model_spectra[0]

        return model_flux, model_spectra, data

    def predict_light_curve(self, obj, count=0, sampling=1., pad=0.):
        # Figure out where to sample the light curve
        min_time = -self.time_window / 2. - pad
        max_time = self.time_window / 2. + pad
        model_times = np.arange(min_time, max_time + sampling, sampling)

        bands = np.arange(len(self.band_map))

        pred_times = np.tile(model_times, len(bands))
        pred_bands = np.repeat(bands, len(model_times))

        model_flux, model_spectra, data = self._predict_single(obj, pred_times,
                                                               pred_bands, count)

        # Reshape model_flux so that it has the shape (batch, band, time)
        model_flux = model_flux.reshape((-1, len(bands), len(model_times)))

        if count == 0:
            # Get rid of the batch index
            model_flux = model_flux[0]

        return model_times, model_flux, data

    def predict_spectrum(self, obj, time, count=0):
        """Predict the spectrum of an object at a given time"""
        pred_times = [time]
        pred_bands = [0]

        model_flux, model_spectra, data = self._predict_single(obj, pred_times,
                                                               pred_bands, count)

        return model_spectra[..., 0]
