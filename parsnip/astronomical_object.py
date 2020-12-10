from matplotlib import pyplot as plt
from tqdm import tqdm
import numpy as np
import pandas as pd
import scipy.stats
import sys

from astropy.table import Table, vstack
from astropy.stats import biweight_location
import avocado
import extinction
import sncosmo

import torch
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F

SIDEREAL_SCALE = 86400. / 86164.0905

class ParsnipObject(avocado.AstronomicalObject):
    """An astronomical object, with metadata and a light curve.

    See `avocado.AstronomicalObject` for details. We add specific functions needed
    for parsnip.
    """
    def resample(self):
        """Resample the light curve by dropping observations and adding noise.

        Returns
        -------
        augmented_object : AstronomicalObject
            A new AstronomicalObject with the resampled light curve.
        """
        metadata = self.metadata.copy()
        obs = self.observations.copy()

        drop_mask = np.ones(len(obs), dtype=bool)

        # Choose a fraction of observations to randomly drop.
        drop_frac = np.random.uniform(0, 0.5)
        drop_mask[np.random.rand(len(obs)) < drop_frac] = False

        # Drop a block of observations. We do several different versions of this with
        # different probabilities. We only consider times where there was significant
        # flux when choosing the times to operate on.
        significant_times = obs['time'][obs['flux'] / obs['flux_error'] > 3.]
        if len(significant_times) > 0.:
            probs = np.array([3., 2., 1., 1.])
            probs /= np.sum(probs)
            mode = np.random.choice(np.arange(len(probs)), p=probs)
            if mode == 0:
                # Don't drop anything.
                pass
            elif mode == 1:
                # Drop all observations after a given one.
                drop_time = np.random.choice(significant_times)
                drop_mask &= obs['time'] < drop_time
            elif mode == 2:
                # Drop all observations before a given one.
                drop_time = np.random.choice(significant_times)
                drop_mask &= obs['time'] > drop_time
            elif mode == 3:
                # Drop all observations in a given window.
                times = np.random.choice(significant_times, 2)
                drop_time_start = min(times)
                drop_time_end = max(times)
                drop_mask &= (
                    (obs['time'] < drop_time_start) | (obs['time'] > drop_time_end)
                )

        # Make sure that we keep at least one observation to keep the code from
        # breaking.
        if not np.any(drop_mask):
            drop_mask[np.random.choice(np.arange(len(drop_mask)))] = True

        # Drop the rejected observations.
        obs = obs[drop_mask]

        # Add noise to the observations.
        if np.random.rand() > 0.5:
            # Figure out the rough scale of our observations.
            s2n_mask = np.argsort(obs['flux'] / obs['flux_error'])[-5:]
            ref_scale = np.clip(np.median(obs['flux'].iloc[s2n_mask]), 0.01, None)

            # Choose an overall scale for the noise from a lognormal distribution.
            noise_scale = ref_scale * np.random.lognormal(-3., 1.)

            # Choose the noise levels for each observation from a lognormal
            # distribution.
            noise_means = np.random.lognormal(np.log(noise_scale) - 0.5, 1., len(obs))

            # Add the noise to the observations.
            obs['flux'] += np.random.normal(0., noise_means)
            obs['flux_error'] = np.sqrt(obs['flux_error']**2 + noise_means**2)

        # Create a new light curve object
        new_obj = AstronomicalObject(metadata, obs)

        return new_obj

    def preprocess(self, autoencoder):
        """Preprocess the light curve and package it as needed for the autoencoder"""
        # Align the observations to a grid in sidereal time.
        self.determine_time_grid()

        # Map each band to its corresponding index.
        obs = self.observations.copy()
        obs['band_indices'] = [autoencoder.band_map.get(i, -1) for i in obs['band']]

        # Cut out any observations that are outside of the window that we are
        # considering.
        grid_times = self.time_to_grid(self.observations['time'])
        pad_window = autoencoder.time_window + 2 * autoencoder.time_pad
        center_time = autoencoder.time_pad + autoencoder.center_time_bin
        time_indices = np.round(grid_times).astype(int) + center_time
        time_mask = (
            (time_indices >= 0)
            & (time_indices < pad_window)
        )
        obs['grid_times'] = grid_times
        obs['time_indices'] = time_indices

        # Correct background levels if desired.
        if autoencoder.correct_background:
            for band in range(len(autoencoder.bands)):
                band_mask = obs['band_indices'] == band
                # Find observations outside of our window.
                outside_obs = obs[~time_mask & band_mask]
                if len(outside_obs) == 0:
                    # No outside observations, don't update the background level.
                    continue

                # Estimate the background level and subtract it.
                background = biweight_location(outside_obs['flux'])
                obs.loc[band_mask, 'flux'] -= background

        # Cut out observations that are in unused bands or outside of the time window.
        band_mask = obs['band_indices'] != -1
        obs = obs[band_mask & time_mask]

        # Correct for Milky Way extinction if desired.
        if autoencoder.correct_mw_extinction:
            band_extinctions = extinction.fm07(
                autoencoder.band_wave_effs, self.metadata['mwebv']
            )
            extinction_scales = 10**(0.4 * band_extinctions[obs['band_indices']])
            obs['flux'] *= extinction_scales
            obs['flux_error'] *= extinction_scales

        # Scale the light curve so that its peak has an amplitude of roughly 1. We use
        # the median time of the five highest signal-to-noise observations to avoid
        # outliers.
        s2n_mask = np.argsort(obs['flux'] / obs['flux_error'])[-5:]
        self.scale = np.median(obs['flux'].iloc[s2n_mask])
        obs['flux'] /= self.scale
        obs['flux_error'] /= self.scale

        # Grid up the flux and flux error.
        # Note that this clobbers any observations that are on the same night, so bin
        # those up before hand if there are a lot of them.
        grid_flux = torch.zeros(len(autoencoder.bands), pad_window)
        grid_fluxerr = torch.zeros(len(autoencoder.bands), pad_window)
        grid_flux[obs['band_indices'].values, obs['time_indices'].values] = \
            torch.FloatTensor(obs['flux'].values)
        grid_fluxerr[obs['band_indices'].values, obs['time_indices'].values] = \
            torch.FloatTensor(obs['flux_error'].values)

        # Build the grid of observations to compare to.
        compare_data = torch.FloatTensor([
            obs['grid_times'].values,
            obs['flux'].values,
            obs['flux_error'].values
        ])
        compare_band_indices = torch.LongTensor(obs['band_indices'].values)

        # Save results
        self.preprocessed_observations = obs
        self.grid_flux = grid_flux
        self.grid_fluxerr = grid_fluxerr
        self.compare_data = compare_data
        self.compare_band_indices = compare_band_indices

    def determine_time_grid(self):
        """Determine the time grid that will be used for the observations."""
        time = self.observations['time']
        sidereal_time = time * SIDEREAL_SCALE

        # Initial guess of the phase. Round everything to 0.1 days, and find the decimal
        # that has the largest count.
        mode, count = scipy.stats.mode(np.round(sidereal_time % 1 + 0.05, 1))
        guess_offset = mode[0] - 0.05

        # Shift everything by the guessed offset
        guess_shift_time = sidereal_time - guess_offset

        # Do a proper estimate of the offset.
        sidereal_offset = guess_offset + np.median((guess_shift_time + 0.5) % 1) - 0.5

        # Shift everything by the final offset estimate.
        shift_time = sidereal_time - sidereal_offset

        # Determine the reference time. We use the median time of the five highest
        # signal-to-noise observations to avoid outliers.
        s2n_mask = np.argsort(self.observations['flux'] /
                              self.observations['flux_error'])[-5:]
        max_time = np.round(np.median(shift_time.iloc[s2n_mask]))

        # Convert back to a reference time in the original units. This reference time
        # corresponds to the reference of the grid in sidereal time.
        self.reference_time = ((max_time + sidereal_offset) / SIDEREAL_SCALE)

    def time_to_grid(self, time):
        """Convert a time in the original units to one on the internal grid"""
        return (time - self.reference_time) * SIDEREAL_SCALE

    def grid_to_time(self, grid_time):
        """Convert a time on the internal grid to a time in the original units"""
        return grid_time / SIDEREAL_SCALE + self.reference_time

    def plot(self):
        plt.figure()

        for band in self.bands:
            band_mask = self.observations['band'] == band
            band_observations = self.observations[band_mask]

            plt.errorbar(
                band_observations['time'],
                band_observations['flux'],
                band_observations['flux_error'],
                fmt='o',
                label=band,
            )

        plt.legend()
        plt.xlabel('Time')
        plt.ylabel('Flux')