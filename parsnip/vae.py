from tqdm import tqdm
import functools
import multiprocessing
import numpy as np
import os
import pandas as pd
import sys

from astropy.cosmology import Planck18
import extinction
import sncosmo

import torch
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F

from .astronomical_object import ParsnipObject
from .utils import frac_to_mag


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


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dilation):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels

        if self.out_channels < self.in_channels:
            raise Exception("out_channels must be >= in_channels.")

        self.conv1 = nn.Conv1d(in_channels, out_channels, 3, dilation=dilation,
                               padding=dilation)
        self.conv2 = nn.Conv1d(out_channels, out_channels, 3,
                               dilation=dilation, padding=dilation)

    def forward(self, x):
        out = self.conv1(x)
        out = F.relu(out)
        out = self.conv2(out)

        # Add back in the input. If it is smaller than the output, pad it first.
        if self.in_channels < self.out_channels:
            pad_size = self.out_channels - self.in_channels
            pad_x = F.pad(x, (0, 0, 0, pad_size))
        else:
            pad_x = x

        # Residual connection
        out = out + pad_x

        out = F.relu(out)

        return out


class Conv1dBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dilation):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels

        self.conv = nn.Conv1d(in_channels, out_channels, 5, dilation=dilation,
                              padding=2*dilation)

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.conv(x)
        out = self.relu(out)

        return out


class GlobalMaxPoolingTime(nn.Module):
    def forward(self, x):
        out, inds = torch.max(x, 2)
        return out


class LightCurveAutoencoder(nn.Module):
    def __init__(
        self,
        name,
        bands,
        device='cuda',
        min_wave=1000.,
        max_wave=11000.,
        spectrum_bins=300,
        max_redshift=4.,
        band_oversampling=51,
        time_window=300,
        time_pad=100,
        time_sigma=20.,
        magsys='ab',
        error_floor=0.01,

        batch_size=128,
        learning_rate=1e-3,
        scheduler_factor=0.5,
        min_learning_rate=1e-5,
        penalty=1e-3,
        augment=True,

        latent_size=3,
        encode_block='residual',
        encode_conv_architecture=[40, 80, 120, 160, 200, 200, 200],
        encode_conv_dilations=[1, 2, 4, 8, 16, 32, 64],
        encode_fc_architecture=[200],
        encode_time_architecture=[200],
        encode_latent_prepool_architecture=[200],
        encode_latent_postpool_architecture=[200],
        decode_architecture=[40, 80, 160],
    ):

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

        self.encode_block = encode_block
        self.encode_conv_architecture = encode_conv_architecture
        self.encode_conv_dilations = encode_conv_dilations
        self.encode_fc_architecture = encode_fc_architecture
        self.encode_time_architecture = encode_time_architecture
        self.encode_latent_prepool_architecture = encode_latent_prepool_architecture
        self.encode_latent_postpool_architecture = encode_latent_postpool_architecture
        self.decode_architecture = decode_architecture

        self.learning_rate = learning_rate
        self.min_learning_rate = min_learning_rate
        self.penalty = penalty
        self.augment = augment

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
            self.optimizer, factor=scheduler_factor, verbose=True
        )

        # Send the model to the desired device
        self.to(self.device, force=True)

    def to(self, device, force=False):
        """Send the model to the given device"""
        new_device = torch.device(device if torch.cuda.is_available() else 'cpu')

        if self.device == new_device and not force:
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

    def preprocess(self, dataset, threads=8, chunksize=64, verbose=True):
        """Preprocess a dataset"""
        if threads == 1:
            iterator = dataset.objects
            if verbose:
                iterator = tqdm(dataset.objects, file=sys.stdout)

            # Run on a single core without multiprocessing
            for obj in iterator:
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
                iterator = p.imap(func, dataset.objects, chunksize=chunksize)
                if verbose:
                    iterator = tqdm(iterator, total=len(dataset.objects),
                                    file=sys.stdout)
                data = list(iterator)

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

        # Extract the data from each object. Numpy is much faster than torch for
        # vector operations currently, so do as much as we can in numpy before
        # converting to torch tensors.

        fluxes = []
        flux_errors = []
        band_indices = []
        time_indices = []
        batch_indices = []
        weights = []

        compare_data = []
        compare_band_indices = []
        redshifts = []
        amp_scales = []

        for idx, obj in enumerate(objects):
            data = obj.preprocess_data

            # Extract the redshift.
            redshifts.append(obj.metadata['redshift'])

            if self.augment:
                time_shift = np.round(np.random.normal(0., self.time_sigma)).astype(int)
                amp_scale = np.exp(np.random.normal(0, 0.5))
            else:
                time_shift = 0
                amp_scale = 1.

            amp_scales.append(amp_scale)

            # Shift and scale the grid data
            obj_times = data['time'] + time_shift
            obj_flux = data['flux'] * amp_scale
            obj_flux_error = data['flux_error'] * amp_scale
            obj_band_indices = data['band_indices']
            obj_time_indices = data['time_indices'] + time_shift
            obj_batch_indices = np.ones_like(obj_band_indices) * idx

            # Mask out data that is outside of the window of the input.
            mask = (obj_time_indices >= 0) & (obj_time_indices < self.time_window)

            if self.augment:
                # Choose a fraction of observations to randomly drop
                drop_frac = np.random.uniform(0, 0.5)
                mask[np.random.rand(len(obj_times)) < drop_frac] = False

            # Apply the mask
            obj_times = obj_times[mask]
            obj_flux = obj_flux[mask]
            obj_flux_error = obj_flux_error[mask]
            obj_band_indices = obj_band_indices[mask]
            obj_time_indices = obj_time_indices[mask]
            obj_batch_indices = obj_batch_indices[mask]

            if self.augment:
                # Add noise to the observations
                if np.random.rand() < 0.5 and len(obj_flux) > 0:
                    # Choose an overall scale for the noise from a lognormal
                    # distribution.
                    noise_scale = np.random.lognormal(-4., 1.)

                    # Choose the noise levels for each observation from a lognormal
                    # distribution.
                    noise_means = np.random.lognormal(np.log(noise_scale), 1.,
                                                      len(obj_flux))

                    # Add the noise to the observations.
                    obj_flux = obj_flux + np.random.normal(0., noise_means)
                    obj_flux_error = np.sqrt(obj_flux_error**2 + noise_means**2)

            # Compute the weight that will be used for comparing the model to the input
            # data. We use the observed error with an error floor.
            obj_weight = 1. / (obj_flux_error**2 + self.error_floor**2)

            # Stack all of the data that will be used for comparisons and convert it to
            # a torch tensor.
            obj_compare_data = torch.FloatTensor(np.vstack([
                obj_times,
                obj_flux,
                obj_flux_error,
                obj_weight,
            ]))
            compare_data.append(obj_compare_data.T)
            compare_band_indices.append(torch.LongTensor(obj_band_indices))

            fluxes.append(obj_flux)
            flux_errors.append(obj_flux_error)
            band_indices.append(obj_band_indices)
            time_indices.append(obj_time_indices)
            batch_indices.append(obj_batch_indices)
            weights.append(obj_weight)

        # Gather the input data.
        fluxes = np.concatenate(fluxes)
        flux_errors = np.concatenate(flux_errors)
        band_indices = np.concatenate(band_indices)
        time_indices = np.concatenate(time_indices)
        batch_indices = np.concatenate(batch_indices)
        weights = np.concatenate(weights)
        redshifts = np.array(redshifts)
        amp_scales = np.array(amp_scales)

        # Build a grid for the input
        grid_flux = np.zeros((len(objects), len(self.bands), self.time_window))
        grid_weights = np.zeros_like(grid_flux)

        grid_flux[batch_indices, band_indices, time_indices] = fluxes
        grid_weights[batch_indices, band_indices, time_indices] = weights

        # Scale the weights so that they are between 0 and 1.
        grid_weights *= self.error_floor**2

        # Stack the input data
        input_data = np.concatenate([
            redshifts[:, None, None].repeat(self.time_window, axis=2),
            grid_flux,
            grid_weights,
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
                    compare_band_indices[0], amp_scales[0])
        else:
            return input_data, compare_data, redshifts, compare_band_indices, amp_scales

    def build_model(self):
        """Build the model"""
        input_size = len(self.bands) * 2 + 1

        if self.encode_block == 'conv1d':
            encode_block = Conv1dBlock
        elif self.encode_block == 'residual':
            encode_block = ResidualBlock
        else:
            raise Exception(f"Unknown block {self.encode_block}.")

        # Encoder architecture. We start with an input of size input_size x
        # self.time_window. We apply a series of convolutional blocks to this that
        # produce outputs that are the same size. The type of block is specified by
        # self.encode_block. Each convolutional block has a dilation that is given by
        # self.encode_conv_dilations.
        if len(self.encode_conv_architecture) != len(self.encode_conv_dilations):
            raise Exception("Layer sizes and dilations must have the same length!")

        encode_layers = []

        # Convolutional layers.
        last_size = input_size
        for layer_size, dilation in zip(self.encode_conv_architecture,
                                        self.encode_conv_dilations):
            encode_layers.append(
                encode_block(last_size, layer_size, dilation)
            )
            last_size = layer_size

        # Fully connected layers for the encoder following the convolution blocks.
        # These are Conv1D layers with a kernel size of 1 that mix within the time
        # indexes.
        for layer_size in self.encode_fc_architecture:
            encode_layers.append(nn.Conv1d(last_size, layer_size, 1))
            encode_layers.append(nn.ReLU())
            last_size = layer_size

        self.encode_layers = nn.Sequential(*encode_layers)

        # Fully connected layers for the time-indexing layer. These are Conv1D layers
        # with a kernel size of 1 that mix within time indexes.
        time_last_size = last_size
        encode_time_layers = []
        for layer_size in self.encode_time_architecture:
            encode_time_layers.append(nn.Conv1d(time_last_size, layer_size, 1))
            encode_time_layers.append(nn.ReLU())
            time_last_size = layer_size

        # Final layer, go down to a single channel with no activation function.
        encode_time_layers.append(nn.Conv1d(time_last_size, 1, 1))
        self.encode_time_layers = nn.Sequential(*encode_time_layers)

        # Fully connected layers to calculate the latent space parameters for the VAE.
        encode_latent_layers = []
        latent_last_size = last_size
        for layer_size in self.encode_latent_prepool_architecture:
            encode_latent_layers.append(nn.Conv1d(latent_last_size, layer_size, 1))
            encode_latent_layers.append(nn.ReLU())
            latent_last_size = layer_size

        # Apply a global max pooling over the time channels.
        encode_latent_layers.append(GlobalMaxPoolingTime())

        # Apply fully connected layers to get the embedding.
        for layer_size in self.encode_latent_postpool_architecture:
            encode_latent_layers.append(nn.Linear(latent_last_size, layer_size))
            encode_latent_layers.append(nn.ReLU())
            latent_last_size = layer_size

        self.encode_latent_layers = nn.Sequential(*encode_latent_layers)

        # Finally, use a last FC layer to get mu and logvar
        self.encode_mu_layer = nn.Linear(latent_last_size, self.latent_size + 1)
        self.encode_logvar_layer = nn.Linear(latent_last_size, self.latent_size + 2)

        # MLP decoder. We start with an input that is the intrinsic latent space + one
        # dimension for time, and output a spectrum of size self.spectrum_bins. We also
        # have hidden layers with sizes given by self.decode_layers. We implement this
        # using a Conv1D layer with a kernel size of 1 for computational reasons so
        # that it decodes multiple spectra for each transient all at the same time, but
        # the decodes are all done independently so this is really an MLP.
        decode_last_size = self.latent_size + 1
        decode_layers = []
        for layer_size in self.decode_architecture:
            decode_layers.append(nn.Conv1d(decode_last_size, layer_size, 1))
            decode_layers.append(nn.Tanh())
            decode_last_size = layer_size

        # Final layer. Use a FC layer to get us to the correct number of bins, and use
        # a softplus activation function to get positive flux.
        decode_layers.append(nn.Conv1d(decode_last_size, self.spectrum_bins, 1))
        decode_layers.append(nn.Softplus())

        self.decode_layers = nn.Sequential(*decode_layers)

    def encode(self, input_data):
        # Apply common encoder blocks
        e = self.encode_layers(input_data)

        # Reference time branch. First, apply additional FC layers to get to an output
        # that has a single channel.
        e_time = self.encode_time_layers(e)

        # Apply the time-indexing layer to calculate the reference time. This is a
        # special layer that is invariant to translations of the input.
        t_vec = torch.nn.functional.softmax(torch.squeeze(e_time, 1), dim=1)
        ref_time_mu = torch.sum(t_vec * self.input_times, 1) / self.time_sigma

        # Latent space branch.
        e_latent = self.encode_latent_layers(e)

        # Predict mu and logvar
        encoding_mu = self.encode_mu_layer(e_latent)
        encoding_logvar = self.encode_logvar_layer(e_latent)

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
        model_spectra = self.decode_layers(stack_encoding)

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

    def reparameterize(self, mu, logvar, sample=True):
        if sample:
            std = torch.exp(0.5*logvar)
            eps = torch.randn_like(std)
            return mu + eps*std
        else:
            return mu

    def sample(self, encoding_mu, encoding_logvar, sample=True):
        sample_encoding = self.reparameterize(encoding_mu, encoding_logvar,
                                              sample=sample)

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

    def forward(self, input_data, compare_data, redshifts, band_indices,
                sample=True):
        encoding_mu, encoding_logvar = self.encode(input_data)

        ref_times, color, encoding = self.sample(encoding_mu, encoding_logvar,
                                                 sample=sample)

        model_spectra, model_flux = self.decode(
            encoding, ref_times, color, compare_data[:, 0], redshifts, band_indices
        )

        flux = compare_data[:, 1]
        weight = compare_data[:, 3]

        # Analytically evaluate the conditional distribution for the amplitude.
        amplitude_mu, amplitude_logvar = self.compute_amplitude(weight, model_flux,
                                                                flux)
        amplitude = self.reparameterize(amplitude_mu, amplitude_logvar, sample=sample)
        model_flux = model_flux * amplitude[:, None]
        model_spectra = model_spectra * amplitude[:, None, None]

        return (encoding_mu, encoding_logvar, amplitude_mu, amplitude_logvar, ref_times,
                color, encoding, amplitude, model_flux, model_spectra)

    def compute_amplitude(self, weight, model_flux, flux):
        num = torch.sum(weight * model_flux * flux, axis=1)
        denom = torch.sum(weight * model_flux * model_flux, axis=1)

        # With augmentation, can very rarely end up with no light curve points. Handle
        # that gracefully by setting the amplitude to 0 with a very large uncertainty.
        denom[denom == 0.] = 1e-5

        amplitude_mu = num / denom
        amplitude_logvar = torch.log(1. / denom)

        return amplitude_mu, amplitude_logvar

    def loss_function(self, compare_data, encoding_mu, encoding_logvar, amplitude_mu,
                      amplitude_logvar, amplitude, model_flux, model_spectra,
                      return_components=False):
        flux = compare_data[:, 1]
        weight = compare_data[:, 3]

        # Reconstruction likelihood
        nll = torch.sum(0.5 * weight * (flux - model_flux)**2)

        # KL divergence
        kld = -0.5 * torch.sum(1 + encoding_logvar - encoding_mu.pow(2) -
                               encoding_logvar.exp())

        # Regularization of spectra
        diff = (
            (model_spectra[:, 1:, :] - model_spectra[:, :-1, :])
            / (model_spectra[:, 1:, :] + model_spectra[:, :-1, :])
        )
        penalty = self.penalty * torch.sum(diff**2)

        # Amplitude probability for the importance sampling integral
        amp_prob = -0.5 * torch.sum((amplitude - amplitude_mu)**2 /
                                    amplitude_logvar.exp())

        if return_components:
            return torch.stack([nll, kld, penalty, amp_prob])
        else:
            return nll + kld + penalty + amp_prob

    def score(self, dataset, rounds=1, return_components=False):
        """Evaluate the loss function on a given dataset.

        Parameters
        ----------
        dataset : `avocado.Dataset`
            Dataset to run on
        rounds : int, optional
            Number of rounds to use for evaluation. VAEs are stochastic, so the loss
            function is not deterministic. By running for multiple rounds, the
            uncertainty on the loss function can be decreased. Default 1.

        Returns
        -------
        loss
            Computed loss function
        """
        self.eval()
        loader = ParsnipDataLoader(dataset, self)

        total_loss = 0
        total_count = 0

        # We score the dataset without augmentation for consistency. Keep track of
        # whether augmentation was turned on so that we can revert to the previous
        # state after we're done.
        augment_state = self.augment

        try:
            self.augment = False

            # Compute the loss
            for round in range(rounds):
                for batch_data in loader:
                    input_data, compare_data, redshifts, band_indices, amp_scales \
                        = batch_data

                    input_data = input_data.to(self.device)
                    compare_data = compare_data.to(self.device)
                    redshifts = redshifts.to(self.device)
                    band_indices = band_indices.to(self.device)

                    encoding_mu, encoding_logvar, amplitude_mu, amplitude_logvar, \
                        ref_times, color, encoding, amplitude, model_flux, \
                        model_spectra = \
                        self(input_data, compare_data, redshifts, band_indices)

                    loss = self.loss_function(compare_data, encoding_mu,
                                              encoding_logvar, amplitude_mu,
                                              amplitude_logvar, amplitude,
                                              model_flux, model_spectra,
                                              return_components)

                    if return_components:
                        total_loss += loss.detach().cpu().numpy()
                    else:
                        total_loss += loss.item()
                    total_count += len(input_data)
        finally:
            # Make sure that we flip the augment switch back to what it started as.
            self.augment = augment_state

        loss = total_loss / total_count

        return loss

    def fit(self, dataset, max_epochs=1000, test_dataset=None):
        """Fit the model to a dataset"""

        # The model is stochastic, so the loss function will have a fair bit of noise.
        # If the dataset is small, we run through several augmentations of it every
        # epoch to get the noise down.
        repeats = int(np.ceil(25000 / len(dataset.objects)))

        while self.epoch < max_epochs:
            loader = ParsnipDataLoader(dataset, self, shuffle=True)
            self.train()
            train_loss = 0
            train_count = 0

            with tqdm(range(loader.num_batches * repeats), file=sys.stdout) as pbar:
                for repeat in range(repeats):
                    # Training step
                    for batch_data in loader:
                        input_data, compare_data, redshifts, band_indices, amp_scales \
                            = batch_data

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

                        total_loss = train_loss / train_count
                        batch_loss = loss.item() / len(input_data)

                        pbar.set_description(
                            f'Epoch {self.epoch:4d}: Loss: {total_loss:8.4f} '
                            f'({batch_loss:8.4f})',
                            refresh=False
                        )
                        pbar.update()

                if test_dataset is not None:
                    # Calculate the test loss
                    test_loss = self.score(test_dataset)
                    pbar.set_description(
                        f'Epoch {self.epoch:4d}: Loss: {total_loss:8.4f}, '
                        f'Test loss: {test_loss:8.4f}',
                    )
                else:
                    pbar.set_description(
                        f'Epoch {self.epoch:4d}: Loss: {total_loss:8.4f}'
                    )

            self.scheduler.step(train_loss)

            # Save the model
            # TODO: Save the model architecture with the model too so that we can
            # recreate it.
            os.makedirs('./models/', exist_ok=True)
            torch.save(self.state_dict(), f'./models/{self.name}.pt')

            # Check if the learning rate is below our threshold, and exit if it is.
            lr = self.optimizer.param_groups[0]['lr']
            if lr < self.min_learning_rate:
                break

            self.epoch += 1

    def load(self):
        """Load the model weights

        TODO: Load the model architecture along with the weights.
        """
        self.load_state_dict(torch.load(f'./models/{self.name}.pt', self.device))

    def predict_dataset(self, dataset, sample=False):
        predictions = []

        loader = ParsnipDataLoader(dataset, self)

        for input_data, compare_data, redshifts, band_indices, amp_scales in loader:
            # Run the data through the model.
            input_data_device = input_data.to(self.device)
            compare_data_device = compare_data.to(self.device)
            redshifts_device = redshifts.to(self.device)
            band_indices_device = band_indices.to(self.device)

            result = self.forward(input_data_device, compare_data_device,
                                  redshifts_device, band_indices_device,
                                  sample=sample)
            result = [i.cpu().detach().numpy() for i in result]
            encoding_mu, encoding_logvar, amplitude_mu, amplitude_logvar, ref_times, \
                color, encoding, amplitude, model_flux, model_spectra = result
            encoding_err = np.sqrt(np.exp(encoding_logvar))

            # Pull out the keys that we care about saving.
            batch_predictions = {
                'reference_time': encoding_mu[:, 0] * self.time_sigma,
                'reference_time_error': encoding_err[:, 0] * self.time_sigma,
                'color': encoding_mu[:, 1],
                'color_error': encoding_err[:, 1],
                'amplitude': amplitude_mu / amp_scales,
                'amplitude_error': np.sqrt(np.exp(amplitude_logvar)) / amp_scales,
            }

            for idx in range(self.latent_size):
                batch_predictions[f's{idx+1}'] = encoding_mu[:, 2 + idx]
                batch_predictions[f's{idx+1}_error'] = encoding_err[:, 2 + idx]

            # Calculate other features
            time, flux, fluxerr, weight = [i.detach().numpy() for i in
                                           compare_data.permute(1, 0, 2)]
            fluxerr_mask = fluxerr == 0
            fluxerr[fluxerr_mask] = -1.

            # Signal-to-noise
            s2n = flux / fluxerr
            s2n[fluxerr_mask] = 0.
            batch_predictions['total_s2n'] = np.sqrt(np.sum(s2n**2, axis=1))

            # Number of observations
            batch_predictions['count'] = np.sum(~fluxerr_mask, axis=1)

            # Number of observations with signal-to-noise above some threshold.
            batch_predictions['count_s2n_3'] = np.sum(s2n > 3, axis=1)
            batch_predictions['count_s2n_5'] = np.sum(s2n > 5, axis=1)

            # Number of observations with signal-to-noise above some threshold in
            # different time windows.
            reference_time = batch_predictions['reference_time'][:, None]
            mask_pre = time < reference_time - 50.
            mask_rise = (time >= reference_time - 50.) & (time < reference_time)
            mask_fall = (time >= reference_time) & (time < reference_time + 50.)
            mask_post = (time >= reference_time + 50.)
            mask_s2n = s2n > 3
            batch_predictions['count_s2n_3_pre'] = np.sum(mask_pre & mask_s2n, axis=1)
            batch_predictions['count_s2n_3_rise'] = np.sum(mask_rise & mask_s2n, axis=1)
            batch_predictions['count_s2n_3_fall'] = np.sum(mask_fall & mask_s2n, axis=1)
            batch_predictions['count_s2n_3_post'] = np.sum(mask_post & mask_s2n, axis=1)

            predictions.append(pd.DataFrame(batch_predictions))

        predictions = pd.concat(predictions, ignore_index=True)

        # Add in the preprocessed scale for the amplitude
        scales = np.array([i.preprocess_data['scale'] for i in dataset.objects])
        predictions['amplitude'] *= scales
        predictions['amplitude_error'] *= scales

        # Merge in the metadata
        predictions.index = dataset.metadata.index
        predictions = pd.concat([dataset.metadata, predictions], axis=1)

        # Estimate the absolute luminosity (assuming a zeropoint of 25).
        amplitudes = predictions['amplitude'].copy()
        amplitude_mask = amplitudes <= 0.
        amplitudes[amplitude_mask] = 1.
        luminosity = (
            -2.5*np.log10(amplitudes)
            + 25.
            - Planck18.distmod(predictions['redshift']).value
        )
        luminosity[amplitude_mask] = np.nan
        predictions['luminosity'] = luminosity

        # Luminosity uncertainty
        frac_diff = predictions['amplitude_error'] / predictions['amplitude']
        frac_diff[amplitude_mask] = 0.5
        frac_diff[frac_diff > 0.5] = 0.5
        int_mag_err = frac_to_mag(frac_diff)
        int_mag_err[amplitude_mask] = -1.
        predictions['luminosity_error'] = int_mag_err

        return predictions

    def predict_dataset_augmented(self, dataset, augments=10, sample=False):
        """Generate predictions for a dataset with augmentation

        This will first generate predictions for the dataset without augmentation,
        and will then generate predictions for the dataset with augmentation the
        given number of times. This returns a dataframe in the same format as
        `predict_dataset`, but with the following additional columns:
        - original_object_id: the original object_id for each augmentation.
        - augmented: True for augmented light curves, False for original ones.

        Parameters
        ----------
        dataset : `avocado.Dataset`
            Dataset to generate predictions for.
        augments : int, optional
            Number of times to augment the dataset, by default 10

        Returns
        -------
        predictions : `pandas.DataFrame`
            Dataframe with one row for each object and columns with each of the
            predicted values.
        """
        # Keep track of whether augmentation was turned on so that we can revert to the
        # previous state after we're done.
        augment_state = self.augment

        try:
            # First pass without augmentation.
            self.augment = False
            pred = self.predict_dataset(dataset, sample=sample)
            pred['original_object_id'] = pred.index
            pred['augmented'] = False

            predictions = [pred]

            # Next passes with augmentation.
            self.augment = True
            for idx in tqdm(range(augments), file=sys.stdout):
                pred = self.predict_dataset(dataset, sample=sample)
                pred['original_object_id'] = pred.index
                pred['augmented'] = True
                pred.index = [i + f'_aug_{idx+1}' for i in pred.index]
                predictions.append(pred)
        finally:
            # Make sure that we flip the augment switch back to what it started as.
            self.augment = augment_state

        predictions = pd.concat(predictions)
        return predictions

    def _predict_single(self, obj, pred_times, pred_bands, count):
        data = self.get_data(obj)
        input_data, compare_data, redshifts, band_indices, amp_scales = data

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
