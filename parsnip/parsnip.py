from tqdm import tqdm
import functools
import multiprocessing
import numpy as np
import os
import sys

from astropy.cosmology import Planck18
import astropy.table
import extinction
import sncosmo
import pkg_resources
import lcdata

import torch
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F
from torch.utils.data import DataLoader

from .light_curve import preprocess_light_curve, grid_to_time, time_to_grid, \
    SIDEREAL_SCALE
from .utils import frac_to_mag, parse_device
from .settings import parse_settings, default_model
from .sncosmo import ParsnipSncosmoSource


class ResidualBlock(nn.Module):
    """1D residual convolutional neural network block

    This module operates on 1D sequences. The input will be padded so that length of the
    sequences is be left unchanged.

    Parameters
    ----------
    in_channels : int
        Number of channels for the input
    out_channels : int
        Number of channels for the output
    dilation : int
        Dilation to use in the convolution
    """
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
    """1D convolutional neural network block

    This module operates on 1D sequences. The input will be padded so that length of the
    sequences is be left unchanged.

    Parameters
    ----------
    in_channels : int
        Number of channels for the input
    out_channels : int
        Number of channels for the output
    dilation : int
        Dilation to use in the convolution
    """
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
    """Time max pooling layer for 1D sequences

    This layer applies global max pooling over all channels to elminate the channel
    dimension while preserving the time dimension.
    """
    def forward(self, x):
        out, inds = torch.max(x, 2)
        return out


class ParsnipModel(nn.Module):
    """Generative model of transient light curves

    This class represents a generative model of transient light curves. Given a set of
    latent variables representing a transient, it can predict the full spectral time
    series of that transient. It can also use variational inference to predict the
    posterior distribution over the latent variables for a given light curve.

    Parameters
    ----------
    path : str
        Path to where the model should be stored on disk.
    bands : List[str]
        Bands that the model uses as input for variational inference
    device : str
        PyTorch device to use for the model
    threads : int
        Number of threads to use
    settings : dict
        Settings for the model. Any settings specified here will override the defaults
        set in settings.py
    ignore_unknown_settings : bool
        If True, ignore any settings that are specified that are unknown. Otherwise,
        raise a KeyError if an unknown setting is specified. By default False.
    """
    def __init__(self, path, bands, device='cpu', threads=8, settings={},
                 ignore_unknown_settings=False):
        super().__init__()

        # Parse settings
        self.settings = parse_settings(bands, settings,
                                       ignore_unknown_settings=ignore_unknown_settings)

        self.path = path
        self.threads = threads

        # Setup the device
        self.device = parse_device(device)
        torch.set_num_threads(self.threads)

        # Setup the bands
        self._setup_band_weights()

        # Setup the color law. We scale this so that the color law has a B-V color of 1,
        # meaning that a coefficient multiplying the color law is the b-v color.
        color_law = extinction.fm07(self.model_wave, 3.1)
        self.color_law = torch.FloatTensor(color_law).to(self.device)

        # Setup the timing
        self.input_times = (torch.arange(self.settings['time_window'],
                                         device=self.device)
                            - self.settings['time_window'] // 2)

        # Build the model
        self._build_model()

        # Set up the training
        self.epoch = 0
        self.optimizer = optim.Adam(self.parameters(),
                                    lr=self.settings['learning_rate'])
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, factor=self.settings['scheduler_factor'], verbose=True
        )

        # Send the model weights to the desired device
        self.to(self.device, force=True)

    def to(self, device, force=False):
        """Send the model to the specified device

        Parameters
        ----------
        device : str
            PyTorch device
        force : bool, optional
            If True, force the model to be sent to the device even if it is there
            already (useful if only parts of the model are there), by default False
        """
        new_device = parse_device(device)

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

    def save(self):
        """Save the model"""
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        torch.save([self.settings, self.state_dict()], self.path)

    def _setup_band_weights(self):
        """Setup the interpolation for the band weights used for photometry"""
        # Build the model in log wavelength
        model_log_wave = np.linspace(np.log10(self.settings['min_wave']),
                                     np.log10(self.settings['max_wave']),
                                     self.settings['spectrum_bins'])
        model_spacing = model_log_wave[1] - model_log_wave[0]

        band_spacing = model_spacing / self.settings['band_oversampling']
        band_max_log_wave = (
            np.log10(self.settings['max_wave'] * (1 + self.settings['max_redshift']))
            + band_spacing
        )

        # Oversampling must be odd.
        assert self.settings['band_oversampling'] % 2 == 1
        pad = (self.settings['band_oversampling'] - 1) // 2
        band_log_wave = np.arange(np.log10(self.settings['min_wave']),
                                  band_max_log_wave, band_spacing)
        band_wave = 10**(band_log_wave)
        band_pad_log_wave = np.arange(
            np.log10(self.settings['min_wave']) - band_spacing * pad,
            band_max_log_wave + band_spacing * pad,
            band_spacing
        )
        band_pad_dwave = (
            10**(band_pad_log_wave + band_spacing / 2.)
            - 10**(band_pad_log_wave - band_spacing / 2.)
        )

        ref = sncosmo.get_magsystem(self.settings['magsys'])

        band_weights = []

        for band_name in self.settings['bands']:
            band = sncosmo.get_bandpass(band_name)
            band_transmission = band(10**(band_pad_log_wave))

            # Convolve the bands to match the sampling of the spectrum.
            band_conv_transmission = np.convolve(
                band_transmission * band_pad_dwave,
                np.ones(self.settings['band_oversampling']),
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
            self.settings['spectrum_bins'] * self.settings['band_oversampling'],
            self.settings['band_oversampling']
        )

        # Save the variables that we need to do interpolation.
        self.band_interpolate_locations = band_interpolate_locations.to(self.device)
        self.band_interpolate_spacing = band_spacing
        self.band_interpolate_weights = torch.FloatTensor(band_weights).to(self.device)
        self.model_wave = 10**(model_log_wave)

    def _calculate_band_weights(self, redshifts):
        """Calculate the band weights for a given set of redshifts

        We have precomputed the weights for each bandpass, so we simply interpolate
        those weights at the desired redshifts. We are working in log-wavelength, so a
        change in redshift just gives us a shift in indices.

        Parameters
        ----------
        redshifts : List[float]
            Redshifts to calculate the band weights at

        Returns
        -------
        `~numpy.ndarray`
            Band weights for each redshift/band combination
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

    def _test_band_weights(self, redshift, source='salt2-extended'):
        """Test the accuracy of the band weights

        We compare sncosmo photometry to the photometry calculated by this class.

        Parameters
        ----------
        redshift : float
            Redshift to evaluate the model at
        source : str, optional
            SNCosmo source to use, by default 'salt2-extended'
        """
        model = sncosmo.Model(source=source)

        # sncosmo photometry
        model.set(z=redshift)
        sncosmo_photometry = model.bandflux(self.settings['bands'], 0., zp=-20.,
                                            zpsys=self.settings['magsys'])

        # parsnip photometry
        model.set(z=0.)
        model_flux = model._flux(0., self.model_wave)[0]
        band_weights = self._calculate_band_weights(
            torch.FloatTensor([redshift]))[0].numpy()
        parsnip_photometry = np.sum(model_flux[:, None] * band_weights, axis=0)

        print(f"z = {redshift}")
        print(f"sncosmo photometry:     {sncosmo_photometry}")
        print(f"parsnip photometry:     {parsnip_photometry}")
        print(f"ratio:                  {parsnip_photometry / sncosmo_photometry}")

    def preprocess(self, dataset, chunksize=64, verbose=True):
        """Preprocess an lcdata dataset

        The preprocessing will be done over multiple threads. Set `ParsnipModel.threads`
        to change how many are used. If the dataset is already preprocessed, then
        nothing will be done and it will be returned as is.

        Parameters
        ----------
        dataset : `~lcdata.Dataset`
            Dataset to preprocess
        chunksize : int, optional
            Number of light curves to process at a time, by default 64
        verbose : bool, optional
            Whether to show a progress bar, by default True

        Returns
        -------
        `~lcdata.Dataset`
            Preprocessed dataset
        """
        import lcdata

        # Check if we were given a preprocessed dataset. We store our preprocessed data
        # as the parsnip_data variable.
        if ('parsnip_preprocessed' in dataset.meta.keys()
                and np.all(dataset.meta['parsnip_preprocessed'])):
            return dataset

        if self.threads == 1:
            iterator = dataset.light_curves
            if verbose:
                iterator = tqdm(dataset.light_curves, file=sys.stdout,
                                desc="Preprocessing dataset")

            # Run on a single core without multiprocessing
            preprocessed_light_curves = []
            for lc in iterator:
                preprocessed_light_curves.append(
                    preprocess_light_curve(lc, self.settings, raise_on_invalid=False)
                )
        else:
            # Run with multiprocessing in multiple threads.
            func = functools.partial(preprocess_light_curve, settings=self.settings,
                                     raise_on_invalid=False)

            with multiprocessing.Pool(self.threads) as p:
                iterator = p.imap(func, dataset.light_curves, chunksize=chunksize)
                if verbose:
                    iterator = tqdm(iterator, total=len(dataset.light_curves),
                                    file=sys.stdout, desc="Preprocessing dataset")
                preprocessed_light_curves = list(iterator)

        # Check if any light curves failed to process
        none_count = 0
        for lc in preprocessed_light_curves:
            if lc is None:
                none_count += 1
        if none_count > 0:
            print(f"WARNING: Rejecting {none_count}/{len(preprocessed_light_curves)} "
                  "light curves. Consider using 'parsnip.load_dataset()' or "
                  "'parsnip.parse_dataset()' to load/parse the dataset and hopefully "
                  "avoid this.")
            preprocessed_light_curves = [i for i in preprocessed_light_curves if i is
                                         not None]

        dataset = lcdata.from_light_curves(preprocessed_light_curves)
        return dataset

    def augment_light_curves(self, light_curves, as_table=True):
        """Augment a set of light curves

        Parameters
        ----------
        light_curves : List[`~astropy.table.Table`]
            List of light curves to augment
        as_table : bool, optional
            Whether to return the light curves as astropy Tables, by default True.
            Constructing new tables is relatively slow, so internally we skip this step
            when training the ParSNIP model.

        Returns
        -------
        List
            Augmented light curves
        """
        # Check if we have a list of light curves or a single one and handle it
        # appropriately.
        if isinstance(light_curves, astropy.table.Table):
            # Single object. Wrap it so that we can process it as an array. We'll unwrap
            # it at the end.
            single = True
            light_curves = [light_curves]
        else:
            single = False

        new_light_curves = np.empty(shape=len(light_curves), dtype=object)

        for idx, lc in enumerate(light_curves):
            # Convert the table to a numpy recarray. This is much faster to work with.
            data = lc.as_array()

            # Make a copy of the metadata to work off of.
            meta = lc.meta.copy(use_cache=not as_table)

            # Randomly drop observations and make a copy of the light curve.
            drop_frac = np.random.uniform(0, 0.5)
            mask = np.random.rand(len(data)) > drop_frac
            data = data[mask]

            # Shift the time randomly.
            time_shift = np.round(
                np.random.normal(0., self.settings['time_sigma'])
            ).astype(int)
            meta['parsnip_reference_time'] += time_shift / SIDEREAL_SCALE
            data['grid_time'] -= time_shift
            data['time_index'] -= time_shift

            # Add noise to the observations
            if np.random.rand() < 0.5 and len(data) > 0:
                # Choose an overall scale for the noise from a lognormal
                # distribution.
                noise_scale = np.random.lognormal(-4., 1.) * meta['parsnip_scale']

                # Choose the noise levels for each observation from a lognormal
                # distribution.
                noise_sigmas = np.random.lognormal(np.log(noise_scale), 1., len(data))

                # Add the noise to the observations.
                noise = np.random.normal(0., noise_sigmas)
                data['flux'] += noise
                data['fluxerr'] = np.sqrt(data['fluxerr']**2 + noise_sigmas**2)

            # Scale the amplitude that we input to the model randomly.
            amp_scale = np.exp(np.random.normal(0, 0.5))
            meta['parsnip_scale'] *= amp_scale

            # Convert back to an astropy Table if desired. This is somewhat slow, so we
            # skip it internally when training the model.
            if as_table:
                new_lc = astropy.table.Table(data, meta=meta)
            else:
                new_lc = (data, meta)

            new_light_curves[idx] = new_lc

        if single:
            return new_light_curves[0]
        else:
            return new_light_curves

    def _get_data(self, light_curves):
        """Extract data needed by ParSNIP from a set of light curves.

        Parameters
        ----------
        light_curves : List[`~astropy.table.Table`]
            Light curves to extract data from

        Returns
        -------
        input_data : `~torch.FloatTensor`
            Data that is used as input to the ParSNIP encoder
        compare_data : `~torch.FloatTensor`
            Data that is used for comparisons with the output of the ParSNIP decoder
        redshifts : `~torch.FloatTensor`
            Redshifts of each light curve
        compare_band_indices : `~torch.LongTensor`
            Band indices for each observation that will be compared
        """
        redshifts = []

        compare_data = []
        compare_band_indices = []

        # Build a grid for the input
        grid_flux = np.zeros((len(light_curves), len(self.settings['bands']),
                              self.settings['time_window']))
        grid_weights = np.zeros_like(grid_flux)

        for idx, lc in enumerate(light_curves):
            # Convert the table to a numpy recarray. This is much faster to work with.
            # For augmentation, we skip creating a Table because that is slow and just
            # keep the recarray. Handle that too.
            if isinstance(lc, astropy.table.Table):
                lc_data = lc.as_array()
                lc_meta = lc.meta
            else:
                lc_data, lc_meta = lc

            # Extract the redshift.
            redshifts.append(lc_meta['redshift'])

            # Mask out observations that are outside of our window.
            mask = (lc_data['time_index'] >= 0) & (lc_data['time_index'] <
                                                   self.settings['time_window'])
            lc_data = lc_data[mask]

            # Scale the flux and fluxerr appropriately. Note that applying the mask
            # makes a copy of the array, so this won't affect the original data.
            lc_data['flux'] /= lc_meta['parsnip_scale']
            lc_data['fluxerr'] /= lc_meta['parsnip_scale']

            # Calculate weights with an error floor included. Note that this typically a
            # very large number. For the comparison this doesn't matter, but for the
            # input we scale it by the error floor so that it becomes a number between 0
            # and 1.
            weights = 1 / (lc_data['fluxerr']**2 + self.settings['error_floor']**2)

            # Fill in the input array.
            grid_flux[idx, lc_data['band_index'], lc_data['time_index']] = \
                lc_data['flux']
            grid_weights[idx, lc_data['band_index'], lc_data['time_index']] = \
                self.settings['error_floor']**2 * weights

            # Stack all of the data that will be used for comparisons and convert it to
            # a torch tensor.
            obj_compare_data = torch.FloatTensor(np.vstack([
                lc_data['grid_time'],
                lc_data['flux'],
                lc_data['fluxerr'],
                weights,
            ]))
            compare_data.append(obj_compare_data.T)
            compare_band_indices.append(torch.LongTensor(lc_data['band_index'].copy()))

        # Gather the input data.
        redshifts = np.array(redshifts)

        # Stack the input data
        if self.settings['input_redshift']:
            input_data = np.concatenate([
                redshifts[:, None, None].repeat(self.settings['time_window'], axis=2),
                grid_flux,
                grid_weights,
            ], axis=1)
        else:
            input_data = np.concatenate([
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

        # Send the arrays to the model's device.
        input_data = input_data.to(self.device)
        compare_data = compare_data.to(self.device)
        redshifts = redshifts.to(self.device)
        compare_band_indices = compare_band_indices.to(self.device)

        return input_data, compare_data, redshifts, compare_band_indices

    def _build_model(self):
        """Build the model"""
        if self.settings['input_redshift']:
            input_size = len(self.settings['bands']) * 2 + 1
        else:
            input_size = len(self.settings['bands']) * 2

        if self.settings['encode_block'] == 'conv1d':
            encode_block = Conv1dBlock
        elif self.settings['encode_block'] == 'residual':
            encode_block = ResidualBlock
        else:
            raise Exception(f"Unknown block {self.settings['encode_block']}.")

        # Encoder architecture.  We start with an input of size input_size x
        # time_window We apply a series of convolutional blocks to this that produce
        # outputs that are the same size.  The type of block is specified by
        # settings['encode_block'].  Each convolutional block has a dilation that is
        # given by settings['encode_conv_dilations'].
        if (len(self.settings['encode_conv_architecture']) !=
                len(self.settings['encode_conv_dilations'])):
            raise Exception("Layer sizes and dilations must have the same length!")

        encode_layers = []

        # Convolutional layers.
        last_size = input_size
        for layer_size, dilation in zip(self.settings['encode_conv_architecture'],
                                        self.settings['encode_conv_dilations']):
            encode_layers.append(
                encode_block(last_size, layer_size, dilation)
            )
            last_size = layer_size

        # Fully connected layers for the encoder following the convolution blocks.
        # These are Conv1D layers with a kernel size of 1 that mix within the time
        # indexes.
        for layer_size in self.settings['encode_fc_architecture']:
            encode_layers.append(nn.Conv1d(last_size, layer_size, 1))
            encode_layers.append(nn.ReLU())
            last_size = layer_size

        self.encode_layers = nn.Sequential(*encode_layers)

        # Fully connected layers for the time-indexing layer. These are Conv1D layers
        # with a kernel size of 1 that mix within time indexes.
        time_last_size = last_size
        encode_time_layers = []
        for layer_size in self.settings['encode_time_architecture']:
            encode_time_layers.append(nn.Conv1d(time_last_size, layer_size, 1))
            encode_time_layers.append(nn.ReLU())
            time_last_size = layer_size

        # Final layer, go down to a single channel with no activation function.
        encode_time_layers.append(nn.Conv1d(time_last_size, 1, 1))
        self.encode_time_layers = nn.Sequential(*encode_time_layers)

        # Fully connected layers to calculate the latent space parameters for the VAE.
        encode_latent_layers = []
        latent_last_size = last_size
        for layer_size in self.settings['encode_latent_prepool_architecture']:
            encode_latent_layers.append(nn.Conv1d(latent_last_size, layer_size, 1))
            encode_latent_layers.append(nn.ReLU())
            latent_last_size = layer_size

        # Apply a global max pooling over the time channels.
        encode_latent_layers.append(GlobalMaxPoolingTime())

        # Apply fully connected layers to get the embedding.
        for layer_size in self.settings['encode_latent_postpool_architecture']:
            encode_latent_layers.append(nn.Linear(latent_last_size, layer_size))
            encode_latent_layers.append(nn.ReLU())
            latent_last_size = layer_size

        self.encode_latent_layers = nn.Sequential(*encode_latent_layers)

        # Finally, use a last FC layer to get mu and logvar
        self.encode_mu_layer = nn.Linear(latent_last_size,
                                         self.settings['latent_size'] + 1)
        self.encode_logvar_layer = nn.Linear(latent_last_size,
                                             self.settings['latent_size'] + 2)

        # MLP decoder. We start with an input that is the intrinsic latent space + one
        # dimension for time, and output a spectrum of size
        # self.settings['spectrum_bins'].  We also have hidden layers with sizes given
        # by self.settings['decode_layers'].  We implement this using a Conv1D layer
        # with a kernel size of 1 for computational reasons so that it decodes multiple
        # spectra for each transient all at the same time, but the decodes are all done
        # independently so this is really an MLP.
        decode_last_size = self.settings['latent_size'] + 1
        decode_layers = []
        for layer_size in self.settings['decode_architecture']:
            decode_layers.append(nn.Conv1d(decode_last_size, layer_size, 1))
            decode_layers.append(nn.Tanh())
            decode_last_size = layer_size

        # Final layer. Use a FC layer to get us to the correct number of bins, and use
        # a softplus activation function to get positive flux.
        decode_layers.append(nn.Conv1d(decode_last_size,
                                       self.settings['spectrum_bins'], 1))
        decode_layers.append(nn.Softplus())

        self.decode_layers = nn.Sequential(*decode_layers)

    def get_data_loader(self, dataset, augment=False, **kwargs):
        """Get a PyTorch DataLoader for an lcdata Dataset

        Parameters
        ----------
        dataset : `~lcdata.Dataset`
            Dataset to load
        augment : bool, optional
            Whether to augment the dataset, by default False

        Returns
        -------
        `~torch.utils.data.DataLoader`
            PyTorch DataLoader for the dataset
        """
        # Preprocess the dataset if it isn't already.
        dataset = self.preprocess(dataset)

        if augment:
            # Reset the metadata caches that we use to speed up augmenting.
            for lc in dataset.light_curves:
                lc.meta.copy(update_cache=True)

            # To speed things up, don't create new astropy.Table objects for the
            # augmented light curves. The `forward` method can handle the result that
            # is returned by `augment_light_curves`.
            collate_fn = functools.partial(self.augment_light_curves, as_table=False)
        else:
            collate_fn = list

        return DataLoader(dataset.light_curves, batch_size=self.settings['batch_size'],
                          collate_fn=collate_fn, **kwargs)

    def encode(self, input_data):
        """Predict the latent variables for a set of light curves

        We use variational inference, and predict the parameters of a posterior
        distribution over the latent space.

        Parameters
        ----------
        input_data : `~torch.FloatTensor`
            Input data representing a set of gridded light curves

        Returns
        -------
        `~torch.FloatTensor`
            Mean predictions for each latent variable
        `~torch.FloatTensor`
            Log-variance predictions for each latent variable
        """
        # Apply common encoder blocks
        e = self.encode_layers(input_data)

        # Reference time branch. First, apply additional FC layers to get to an output
        # that has a single channel.
        e_time = self.encode_time_layers(e)

        # Apply the time-indexing layer to calculate the reference time. This is a
        # special layer that is invariant to translations of the input.
        t_vec = torch.nn.functional.softmax(torch.squeeze(e_time, 1), dim=1)
        ref_time_mu = (
            torch.sum(t_vec * self.input_times, 1)
            / self.settings['time_sigma']
        )

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

    def decode_spectra(self, encoding, phases, color, amplitude=None):
        """Predict the spectra at a given set of latent variables

        Parameters
        ----------
        encoding : `~torch.FloatTensor`
            Coordinates in the ParSNIP intrinsic latent space for each light curve
        phases : `~torch.FloatTensor`
            Phases to decode each light curve at
        color : `~torch.FloatTensor`
            Color of each light curve
        amplitude : `~torch.FloatTensor`, optional
            Amplitude to scale each light curve by, by default no scaling will be
            applied.

        Returns
        -------
        `~torch.FloatTensor`
            Predicted spectra
        """
        scale_phases = phases / (self.settings['time_window'] // 2)

        repeat_encoding = encoding[:, :, None].expand((-1, -1, scale_phases.shape[1]))
        stack_encoding = torch.cat([repeat_encoding, scale_phases[:, None, :]], 1)

        # Apply intrinsic decoder
        model_spectra = self.decode_layers(stack_encoding)

        if color is not None:
            # Apply colors
            apply_colors = 10**(-0.4 * color[:, None] * self.color_law[None, :])
            model_spectra = model_spectra * apply_colors[..., None]

        if amplitude is not None:
            # Apply amplitude
            model_spectra = model_spectra * amplitude[:, None, None]

        return model_spectra

    def decode(self, encoding, ref_times, color, times, redshifts, band_indices,
               amplitude=None):
        """Predict the light curves for a given set of latent variables

        Parameters
        ----------
        encoding : `~torch.FloatTensor`
            Coordinates in the ParSNIP intrinsic latent space for each light curve
        ref_times : `~torch.FloatTensor`
            Reference time for each light curve
        color : `~torch.FloatTensor`
            Color of each light curve
        times : `~torch.FloatTensor`
            Times to predict each light curve at
        redshifts : `~torch.FloatTensor`
            Redshift of each light curve
        band_indices : `~torch.LongTensor`
            Band indices for each observation
        amplitude : `~torch.FloatTensor`, optional
            Amplitude to scale each light curve by, by default no scaling will be
            applied

        Returns
        -------
        `~torch.FloatTensor`
            Model spectra
        `~torch.FloatTensor`
            Model photometry
        """
        phases = (
            (times - ref_times[:, None])
            / (1 + redshifts[:, None])
        )

        # Generate the restframe spectra
        model_spectra = self.decode_spectra(encoding, phases, color, amplitude)

        # Figure out the weights for each band
        band_weights = self._calculate_band_weights(redshifts)
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

        return model_spectra, model_flux

    def _reparameterize(self, mu, logvar, sample=True):
        if sample:
            std = torch.exp(0.5*logvar)
            eps = torch.randn_like(std)
            return mu + eps*std
        else:
            return mu

    def _sample(self, encoding_mu, encoding_logvar, sample=True):
        sample_encoding = self._reparameterize(encoding_mu, encoding_logvar,
                                               sample=sample)

        time_sigma = self.settings['time_sigma']
        color_sigma = self.settings['color_sigma']

        # Rescale variables
        ref_times = sample_encoding[:, 0] * time_sigma
        color = sample_encoding[:, 1] * color_sigma
        encoding = sample_encoding[:, 2:]

        # Constrain the color and reference time so that things don't go to crazy values
        # and throw everything off with floating point precision errors. This will not
        # be a concern for a properly trained model, but things can go wrong early in
        # the training at high learning rates.
        ref_times = torch.clamp(ref_times, -10. * time_sigma, 10. * time_sigma)
        color = torch.clamp(color, -10. * color_sigma, 10. * color_sigma)

        return ref_times, color, encoding

    def forward(self, light_curves, sample=True, to_numpy=False):
        """Run a set of light curves through the full ParSNIP model

        We use variational inference to predict the latent representation of each light
        curve, and we then use the generative model to predict the light curves for
        those representations.

        Parameters
        ----------
        light_curves : List[`~astropy.table.Table`]
            List of light curves
        sample : bool, optional
            If True (default), sample from the posterior distribution. If False, use the
            MAP.
        to_numpy : bool, optional
            Whether to convert the outputs to numpy arrays, by default False

        Returns
        -------
        dict
            Result dictionary. If to_numpy is True, all of the elements will be numpy
            arrays. Otherwise, they will be PyTorch tensors on the model's device.
        """
        # Extract the data that we need and move it to the right device.
        input_data, compare_data, redshifts, band_indices = self._get_data(light_curves)

        # Encode the light curves.
        encoding_mu, encoding_logvar = self.encode(input_data)

        # Sample from the latent space.
        ref_times, color, encoding = self._sample(encoding_mu, encoding_logvar,
                                                  sample=sample)

        # Decode the light curves
        model_spectra, model_flux = self.decode(
            encoding, ref_times, color, compare_data[:, 0], redshifts, band_indices
        )

        flux = compare_data[:, 1]
        weight = compare_data[:, 3]

        # Analytically evaluate the conditional distribution for the amplitude and
        # sample from it.
        amplitude_mu, amplitude_logvar = self._compute_amplitude(weight, model_flux,
                                                                 flux)
        amplitude = self._reparameterize(amplitude_mu, amplitude_logvar, sample=sample)
        model_flux = model_flux * amplitude[:, None]
        model_spectra = model_spectra * amplitude[:, None, None]

        result = {
            'ref_times': ref_times,
            'color': color,
            'encoding': encoding,
            'amplitude': amplitude,
            'time': compare_data[:, 0],
            'obs_flux': compare_data[:, 1],
            'obs_fluxerr': compare_data[:, 2],
            'obs_weight': compare_data[:, 3],
            'band_indices': band_indices,
            'model_flux': model_flux,
            'model_spectra': model_spectra,
            'encoding_mu': encoding_mu,
            'encoding_logvar': encoding_logvar,
            'amplitude_mu': amplitude_mu,
            'amplitude_logvar': amplitude_logvar,
        }

        if to_numpy:
            result = {k: v.detach().cpu().numpy() for k, v in result.items()}

        return result

    def _compute_amplitude(self, weight, model_flux, flux):
        num = torch.sum(weight * model_flux * flux, axis=1)
        denom = torch.sum(weight * model_flux * model_flux, axis=1)

        # With augmentation, can very rarely end up with no light curve points. Handle
        # that gracefully by setting the amplitude to 0 with a very large uncertainty.
        denom[denom == 0.] = 1e-5

        amplitude_mu = num / denom
        amplitude_logvar = torch.log(1. / denom)

        return amplitude_mu, amplitude_logvar

    def loss_function(self, result, return_components=False):
        """Compute the loss function for a set of light curves

        Parameters
        ----------
        result : dict
            Output of `~ParsnipModel.forward`
        return_components : bool, optional
            Whether to return the individual parts of the loss function, by default
            False

        Returns
        -------
        float or `~torch.FloatTensor`
            If return_components is False, return a single value representing the loss
            function for a set of light curves. If return_components is True, then we
            return a set of four values representing the negative log likelihood, the KL
            divergence, the regularization penalty, and the amplitude probability.
        """
        # Reconstruction likelihood
        nll = torch.sum(0.5 * result['obs_weight'] *
                        (result['obs_flux'] - result['model_flux'])**2)

        # KL divergence
        kld = -0.5 * torch.sum(1 + result['encoding_logvar'] -
                               result['encoding_mu'].pow(2) -
                               result['encoding_logvar'].exp())

        # Regularization of spectra
        diff = (
            (result['model_spectra'][:, 1:, :] - result['model_spectra'][:, :-1, :])
            / (result['model_spectra'][:, 1:, :] + result['model_spectra'][:, :-1, :])
        )
        penalty = self.settings['penalty'] * torch.sum(diff**2)

        # Amplitude probability for the importance sampling integral
        amp_prob = -0.5 * torch.sum((result['amplitude'] - result['amplitude_mu'])**2 /
                                    result['amplitude_logvar'].exp())

        if return_components:
            return torch.stack([nll, kld, penalty, amp_prob])
        else:
            return nll + kld + penalty + amp_prob

    def score(self, dataset, rounds=1, return_components=False):
        """Evaluate the loss function on a given dataset.

        Parameters
        ----------
        dataset : `~lcdata.Dataset`
            Dataset to run on
        rounds : int, optional
            Number of rounds to use for evaluation. VAEs are stochastic, so the loss
            function is not deterministic. By running for multiple rounds, the
            uncertainty on the loss function can be decreased. Default 1.
        return_components : bool, optional
            Whether to return the individual parts of the loss function, by default
            False. See `~ParsnipModel.loss_function` for details.

        Returns
        -------
        loss
            Computed loss function
        """
        self.eval()

        total_loss = 0
        total_count = 0

        loader = self.get_data_loader(dataset)

        # Compute the loss
        for round in range(rounds):
            for batch_lcs in loader:
                result = self.forward(batch_lcs)
                loss = self.loss_function(result, return_components)

                if return_components:
                    total_loss += loss.detach().cpu().numpy()
                else:
                    total_loss += loss.item()
                total_count += len(batch_lcs)

        loss = total_loss / total_count

        return loss

    def fit(self, dataset, max_epochs=1000, augment=True, test_dataset=None):
        """Fit the model to a dataset

        Parameters
        ----------
        dataset : `~lcdata.Dataset`
            Dataset to fit to
        max_epochs : int, optional
            Maximum number of epochs, by default 1000
        augment : bool, optional
            Whether to use augmentation, by default True
        test_dataset : `~lcdata.Dataset`, optional
            Test dataset that will be scored at the end of each epoch, by default None
        """
        # The model is stochastic, so the loss function will have a fair bit of noise.
        # If the dataset is small, we run through several augmentations of it every
        # epoch to get the noise down.
        repeats = int(np.ceil(25000 / len(dataset)))

        loader = self.get_data_loader(dataset, augment=augment, shuffle=True)

        if test_dataset is not None:
            test_dataset = self.preprocess(test_dataset)

        while self.epoch < max_epochs:
            self.train()
            train_loss = 0
            train_count = 0

            with tqdm(range(len(loader) * repeats), file=sys.stdout) as pbar:
                for repeat in range(repeats):
                    # Training step
                    for batch_lcs in loader:
                        self.optimizer.zero_grad()
                        result = self.forward(batch_lcs)

                        loss = self.loss_function(result)

                        loss.backward()
                        train_loss += loss.item()
                        self.optimizer.step()

                        train_count += len(batch_lcs)

                        total_loss = train_loss / train_count
                        batch_loss = loss.item() / len(batch_lcs)

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

            # Checkpoint and save the model
            self.save()

            # Check if the learning rate is below our threshold, and exit if it is.
            lr = self.optimizer.param_groups[0]['lr']
            if lr < self.settings['min_learning_rate']:
                break

            self.epoch += 1

    def predict(self, light_curves, augment=False):
        """Generate predictions for a light curve or set of light curves.

        Parameters
        ----------
        light_curves : `~astropy.table.Table` or List[`~astropy.table.Table`]
            Light curve(s) to generate predictions for.
        augment : bool, optional
            Whether to augment the light curve(s), by default False

        Returns
        -------
        `~astropy.table.Table` or dict
            Table (for multiple light curves) or dict (for a single light curve)
            containing the predictions.
        """
        # Check if we have a list of light curves or a single one and handle it
        # appropriately.
        if isinstance(light_curves, astropy.table.Table):
            # Single object. Wrap it so that we can process it as an array. We'll unwrap
            # it at the end.
            single = True
            light_curves = [light_curves]
        else:
            single = False

        # Wrap the light curves in an lcdata dataset and use that to process them.
        dataset = lcdata.from_light_curves(light_curves)
        predictions = self.predict_dataset(dataset, augment=augment)

        if single:
            return dict(zip(predictions[0].keys(), predictions[0].values()))
        else:
            return predictions

    def predict_dataset(self, dataset, augment=False):
        """Generate predictions for a dataset

        Parameters
        ----------
        dataset : `~lcdata.Dataset`
            Dataset to generate predictions for.
        augment : bool, optional
            Whether to perform augmentation, False by default.

        Returns
        -------
        predictions : `~astropy.table.Table`
            astropy Table with one row for each light curve and columns with each of the
            predicted values.
        """
        predictions = []

        dataset = self.preprocess(dataset, verbose=len(dataset) > 100)
        loader = self.get_data_loader(dataset, augment=augment)

        for batch_lcs in loader:
            # Run the data through the model.
            result = self.forward(batch_lcs, to_numpy=True, sample=False)

            # Pull out the reference time and reference scale. Note that if we are
            # working with an augmented dataset, get_data_loader doesn't construct a
            # full astropy Table to save time. Handle either case.
            parsnip_reference_time = []
            parsnip_scale = []
            for lc in batch_lcs:
                if isinstance(lc, astropy.table.Table):
                    lc_meta = lc.meta
                else:
                    lc_data, lc_meta = lc
                parsnip_reference_time.append(lc_meta['parsnip_reference_time'])
                parsnip_scale.append(lc_meta['parsnip_scale'])
            parsnip_reference_time = np.array(parsnip_reference_time)
            parsnip_scale = np.array(parsnip_scale)

            encoding_mu = result['encoding_mu']
            encoding_err = np.sqrt(np.exp(result['encoding_logvar']))

            # Update the reference time.
            reference_time_offset = (
                encoding_mu[:, 0] * self.settings['time_sigma'] / SIDEREAL_SCALE
            )
            reference_time = parsnip_reference_time + reference_time_offset
            reference_time_error = (
                encoding_err[:, 0] * self.settings['time_sigma'] / SIDEREAL_SCALE
            )

            amplitude_mu = result['amplitude_mu'] * parsnip_scale
            amplitude_error = (
                np.sqrt(np.exp(result['amplitude_logvar'])) * parsnip_scale
            )

            # Pull out the keys that we care about saving.
            batch_predictions = {
                'reference_time': reference_time,
                'reference_time_error': reference_time_error,
                'color': encoding_mu[:, 1] * self.settings['color_sigma'],
                'color_error': encoding_err[:, 1] * self.settings['color_sigma'],
                'amplitude': amplitude_mu,
                'amplitude_error': amplitude_error,
            }

            for idx in range(self.settings['latent_size']):
                batch_predictions[f's{idx+1}'] = encoding_mu[:, 2 + idx]
                batch_predictions[f's{idx+1}_error'] = encoding_err[:, 2 + idx]

            # Calculate other useful features.
            time = result['time']
            obs_flux = result['obs_flux']
            obs_fluxerr = result['obs_fluxerr']
            model_flux = result['model_flux']
            fluxerr_mask = obs_fluxerr == 0
            obs_fluxerr[fluxerr_mask] = -1.

            # Signal-to-noise
            s2n = obs_flux / obs_fluxerr
            s2n[fluxerr_mask] = 0.
            batch_predictions['total_s2n'] = np.sqrt(np.sum(s2n**2, axis=1))

            # Number of observations
            batch_predictions['count'] = np.sum(~fluxerr_mask, axis=1)

            # Number of observations with signal-to-noise above some threshold.
            batch_predictions['count_s2n_3'] = np.sum(s2n > 3, axis=1)
            batch_predictions['count_s2n_5'] = np.sum(s2n > 5, axis=1)

            # Number of observations with signal-to-noise above some threshold in
            # different time windows.
            compare_time = reference_time_offset[:, None]
            mask_pre = time < compare_time - 50.
            mask_rise = (time >= compare_time - 50.) & (time < compare_time)
            mask_fall = (time >= compare_time) & (time < compare_time + 50.)
            mask_post = (time >= compare_time + 50.)
            mask_s2n = s2n > 3
            batch_predictions['count_s2n_3_pre'] = np.sum(mask_pre & mask_s2n, axis=1)
            batch_predictions['count_s2n_3_rise'] = np.sum(mask_rise & mask_s2n, axis=1)
            batch_predictions['count_s2n_3_fall'] = np.sum(mask_fall & mask_s2n, axis=1)
            batch_predictions['count_s2n_3_post'] = np.sum(mask_post & mask_s2n, axis=1)

            # Chi-square
            all_chisq = (obs_flux - model_flux)**2 / obs_fluxerr**2
            all_chisq[fluxerr_mask] = 0.
            batch_predictions['model_chisq'] = np.sum(all_chisq, axis=1)
            batch_predictions['model_dof'] = (
                batch_predictions['count']
                - 3             # amplitude, color, reference time
                - self.settings['latent_size']
            )

            predictions.append(astropy.table.Table(batch_predictions))

        predictions = astropy.table.vstack(predictions, 'exact')

        # Drop any old predictions from the metadata, and merge it in.
        meta = dataset.meta.copy()
        common_columns = set(predictions.colnames) & set(meta.colnames)
        meta.remove_columns(common_columns)
        predictions = astropy.table.hstack([meta, predictions], 'exact')

        # Estimate the absolute luminosity (assuming a zeropoint of 25).
        # Figure out which light curves we can calculate the luminosity for.
        amplitudes = predictions['amplitude'].copy()
        amplitude_mask = amplitudes > 0.
        redshifts = predictions['redshift'].copy()
        redshift_mask = redshifts > 0.
        frac_diff = predictions['amplitude_error'] / amplitudes
        amplitude_error_mask = frac_diff < 1.
        luminosity_mask = amplitude_mask & redshift_mask & amplitude_error_mask

        # Mask out invalid data for luminosities
        redshifts[~luminosity_mask] = 1.
        amplitudes[~luminosity_mask] = 1.
        frac_diff[~luminosity_mask] = 0.5

        luminosity = (
            -2.5*np.log10(amplitudes)
            + 25.
            - Planck18.distmod(redshifts).value
        )
        luminosity[~luminosity_mask] = np.nan
        predictions['luminosity'] = luminosity

        # Luminosity uncertainty
        int_mag_err = frac_to_mag(frac_diff)
        int_mag_err[~luminosity_mask] = np.nan
        predictions['luminosity_error'] = int_mag_err

        # Remove the processing flag.
        del predictions['parsnip_preprocessed']

        return predictions

    def predict_dataset_augmented(self, dataset, augments=10):
        """Generate predictions for a dataset with augmentation

        This will first generate predictions for the dataset without augmentation,
        and will then generate predictions for the dataset with augmentation the
        given number of times. This returns a dataframe in the same format as
        `~predict_dataset`, but with the following additional columns:
        - original_object_id: the original object_id for each augmentation.
        - augmented: True for augmented light curves, False for original ones.

        Parameters
        ----------
        dataset : `~lcdata.Dataset`
            Dataset to generate predictions for.
        augments : int, optional
            Number of times to augment the dataset, by default 10

        Returns
        -------
        predictions : `~astropy.table.Table`
            astropy Table with one row for each light curve and columns with each of the
            predicted values.
        """
        # First pass without augmentation.
        pred = self.predict_dataset(dataset)
        pred['original_object_id'] = pred['object_id']
        pred['augmented'] = False

        predictions = [pred]

        # Next passes with augmentation.
        for idx in tqdm(range(augments), file=sys.stdout):
            pred = self.predict_dataset(dataset, augment=True)
            pred['original_object_id'] = pred['object_id']
            pred['augmented'] = True
            pred['object_id'] = [i + f'_aug_{idx+1}' for i in pred['object_id']]
            predictions.append(pred)

        predictions = astropy.table.vstack(predictions, 'exact')
        return predictions

    def _predict_time_series(self, lc, pred_times, pred_bands, sample, count):
        # Convert given times to our internal times.
        grid_times = time_to_grid(pred_times, lc.meta['parsnip_reference_time'])

        grid_times = torch.FloatTensor(grid_times)[None, :].to(self.device)
        pred_bands = torch.LongTensor(pred_bands)[None, :].to(self.device)

        if count is not None:
            # Predict multiple light curves
            light_curves = [lc] * count
            grid_times = grid_times.repeat(count, 1)
            pred_bands = pred_bands.repeat(count, 1)
        else:
            light_curves = [lc]

        # Sample VAE parameters
        result = self.forward(light_curves, sample)

        # Do the predictions
        redshifts = torch.FloatTensor([i.meta['redshift'] for i in light_curves],
                                      device=self.device)
        model_spectra, model_flux = self.decode(
            result['encoding'],
            result['ref_times'],
            result['color'],
            grid_times,
            redshifts,
            pred_bands,
            result['amplitude'],
        )

        model_flux = model_flux.cpu().detach().numpy()
        model_spectra = model_spectra.cpu().detach().numpy()

        if count is None:
            # Get rid of the batch index
            model_flux = model_flux[0]
            model_spectra = model_spectra[0]

        cpu_result = {k: v.detach().cpu().numpy() for k, v in result.items()}

        # Scale everything to the original light curve scale.
        model_flux *= lc.meta['parsnip_scale']
        model_spectra *= lc.meta['parsnip_scale']

        return model_flux, model_spectra, cpu_result

    def predict_light_curve(self, light_curve, sample=False, count=None, sampling=1.,
                            pad=50.):
        """Predict the flux of a light curve on a grid

        Parameters
        ----------
        light_curve : `~astropy.table.Table`
            Light curve to predict
        sample : bool, optional
            If True, sample from the latent variable posteriors. Otherwise,
            use the MAP. By default False.
        count : int, optional
            Number of light curves to predict, by default None (single prediction)
        sampling : int, optional
            Grid sampling in days, by default 1.
        pad : int, optional
            Number of days before and after the light curve observations to predict the
            light curve at, by default 50.

        Returns
        -------
        `~numpy.ndarray`
            Times that the model was sampled at
        `~numpy.ndarray`
            Flux of the model in each band
        `~numpy.ndarray`
            Model result from ParsnipModel.forward
        """
        # Figure out where to sample the light curve
        min_time = np.min(light_curve['time']) - pad
        max_time = np.max(light_curve['time']) + pad
        model_times = np.arange(min_time, max_time + sampling, sampling)

        band_indices = np.arange(len(self.settings['bands']))

        pred_times = np.tile(model_times, len(band_indices))
        pred_bands = np.repeat(band_indices, len(model_times))

        model_flux, model_spectra, model_result = self._predict_time_series(
            light_curve, pred_times, pred_bands, sample, count
        )

        # Reshape model_flux so that it has the shape (batch, band, time)
        model_flux = model_flux.reshape((-1, len(band_indices), len(model_times)))

        if count == 0:
            # Get rid of the batch index
            model_flux = model_flux[0]

        return model_times, model_flux, model_result

    def predict_spectrum(self, light_curve, time, sample=False, count=None):
        """Predict the spectrum of a light curve at a given time

        Parameters
        ----------
        light_curve : `~astropy.table.Table`
            Light curve
        time : float
            Time to predict the spectrum at
        sample : bool, optional
            If True, sample from the latent variable posteriors. Otherwise,
            use the MAP. By default False.
        count : int, optional
            Number of spectra to predict, by default None (single prediction)

        Returns
        -------
        `~numpy.ndarray`
            Predicted spectrum at the wavelengths specified by
            `~ParsnipModel.model_wave`
        """
        pred_times = [time]
        pred_bands = [0]

        model_flux, model_spectra, model_result = self._predict_time_series(
            light_curve, pred_times, pred_bands, sample, count
        )

        return model_spectra[..., 0]

    def predict_sncosmo(self, light_curve, sample=False):
        """Package the predictions for a light curve as an sncosmo model

        This method performs variational inference on a light curve to predict its
        latent representation. It then initializes an SNCosmo model with that
        representation.

        Parameters
        ----------
        light_curve : `~astropy.table.Table`
            Light curve
        sample : bool, optional
            If True, sample from the latent variable posteriors. Otherwise,
            use the MAP. By default False.

        Returns
        -------
        `~ParsnipSncosmoModel`
            SNCosmo model initialized with the light curve's predicted latent
            representation
        """
        if not light_curve.meta.get('parsnip_preprocessed', False):
            light_curve = preprocess_light_curve(light_curve, self.settings)

        # Run through the model to predict parameters.
        result = self.forward([light_curve], sample=sample, to_numpy=True)

        # Build the sncosmo model.
        model = sncosmo.Model(source=ParsnipSncosmoSource(self))

        meta = light_curve.meta
        model['z'] = meta['redshift']
        model['t0'] = grid_to_time(result['ref_times'][0],
                                   meta['parsnip_reference_time'])
        model['color'] = result['color'][0]

        # Note: ZP of amplitude is 25, and we use an internal offset of 20 for building
        # the model so that things are close to 1. Combined, that means that we need to
        # apply an offset of 45 mag when calculating the amplitude for sncosmo.
        model['amplitude'] = (
            light_curve.meta['parsnip_scale'] * result['amplitude'][0]
            * 10**(-0.4 * 45)
        )

        for i in range(self.settings['latent_size']):
            model[f's{i+1}'] = result['encoding'][0, i]

        return model


def load_model(path=None, device='cpu', threads=8):
    """Load a ParSNIP model.

    Parameters
    ----------
    path : str, optional
        Path to the model on disk, or name of a model. If not specified, the
        default_model specified in settings.py is loaded.
    device : str, optional
        Torch device to load the model to, by default 'cpu'
    threads : int, optional
        Number of threads to use, by default 8

    Returns
    -------
    `~ParsnipModel`
        Loaded model
    """
    if path is None:
        path = default_model
        print(f"Loading default ParSNIP model '{path}'")

    # Figure out if we were given the path to a model or a built-in model.
    if '.' not in path:
        # We were given the name of a built-in model.
        resource_path = f'models/{path}.pt'
        full_path = pkg_resources.resource_filename('parsnip', resource_path)
        if not os.path.exists(full_path):
            raise ValueError(f"No built-in model named '{path}'")
        path = full_path

    # Load the model data
    use_device = parse_device(device)
    settings, state_dict = torch.load(path, use_device)

    # Instantiate the model
    model = ParsnipModel(path, settings['bands'], use_device, threads, settings)
    model.load_state_dict(state_dict)

    return model
