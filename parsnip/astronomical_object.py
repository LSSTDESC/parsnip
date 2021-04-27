import numpy as np
import scipy.stats

from astropy.stats import biweight_location

SIDEREAL_SCALE = 86400. / 86164.0905


def resample(astronomical_object):
    """Resample a light curve by dropping observations and adding noise.

    Parameters
    ----------
    astronomical_object : AstronomicalObject
        The light curve to resample

    Returns
    -------
    augmented_object : AstronomicalObject
        A new AstronomicalObject with the resampled light curve.
    """
    metadata = astronomical_object.metadata.copy()
    obs = astronomical_object.observations.copy()

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
    new_obj = type(astronomical_object)(metadata, obs)

    return new_obj


def _determine_time_grid(astronomical_object):
    """Determine the time grid that will be used for the observations."""
    obs = astronomical_object.observations
    time = obs['time']
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

    # Determine the reference time for the light curve.
    # This is tricky to do right. We want to roughly estimate where the "peak" of
    # the light curve is. Oftentimes we see low signal-to-noise observations that
    # are much larger than the peak flux though. This algorithm tries to find a
    # nice balance to handle that.

    # Find the five highest signal-to-noise observations
    s2n = obs['flux'] / obs['flux_error']
    s2n_mask = np.argsort(s2n)[-5:]

    # If we have very few observations, only keep the ones above signal-to-noise of
    # 5 if possible. Sometimes we only have a single point on the rise so far, so
    # we don't want to include a bunch of bad observations in our determination of
    # the time.
    s2n_mask_2 = s2n.iloc[s2n_mask] > 5.
    if np.any(s2n_mask_2):
        cut_times = shift_time.iloc[s2n_mask][s2n_mask_2]
    else:
        # No observations with signal-to-noise above 5. Just use whatever we
        # have...
        cut_times = shift_time.iloc[s2n_mask]

    max_time = np.round(np.median(cut_times))

    # Convert back to a reference time in the original units. This reference time
    # corresponds to the reference of the grid in sidereal time.
    reference_time = ((max_time + sidereal_offset) / SIDEREAL_SCALE)
    return reference_time


def time_to_grid(time, reference_time):
    """Convert a time in the original units to one on the internal grid"""
    return (time - reference_time) * SIDEREAL_SCALE


def grid_to_time(grid_time, reference_time):
    """Convert a time on the internal grid to a time in the original units"""
    return grid_time / SIDEREAL_SCALE + reference_time


def preprocess_astronomical_object(astronomical_object, settings):
    """Preprocess the light curve and package it as needed for ParSNIP"""
    # Align the observations to a grid in sidereal time.
    reference_time = _determine_time_grid(astronomical_object)

    # Map each band to its corresponding index.
    obs = astronomical_object.observations.copy()
    band_map = {j: i for i, j in enumerate(settings['bands'])}
    obs['band_indices'] = [band_map.get(i, -1) for i in obs['band']]

    # Cut out any observations that are outside of the window that we are
    # considering.
    grid_times = time_to_grid(obs['time'], reference_time)
    time_indices = np.round(grid_times).astype(int) + settings['time_window'] // 2
    time_mask = (
        (time_indices >= -settings['time_pad'])
        & (time_indices < settings['time_window'] + settings['time_pad'])
    )
    obs['grid_times'] = grid_times
    obs['time_indices'] = time_indices

    # Correct background levels for bands that need it.
    for band_idx, do_correction in enumerate(settings['band_correct_background']):
        if not do_correction:
            continue

        band_mask = obs['band_indices'] == band_idx
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
    band_extinctions = (
        settings['band_mw_extinctions'] * astronomical_object.metadata['mwebv']
    )
    extinction_scales = 10**(0.4 * band_extinctions[obs['band_indices']])
    obs['flux'] *= extinction_scales
    obs['flux_error'] *= extinction_scales

    # Scale the light curve so that its peak has an amplitude of roughly 1. We use
    # the brightest observation with signal-to-noise above 5 if there is one, or
    # simply the brightest observation otherwise.
    s2n = obs['flux'] / obs['flux_error']
    s2n_mask = s2n > 5.
    if np.any(s2n_mask):
        scale = np.max(obs['flux'][s2n_mask])
    else:
        scale = np.max(obs['flux'])

    obs['flux'] /= scale
    obs['flux_error'] /= scale

    # Save the result
    preprocess_data = {
        'reference_time': reference_time,
        'scale': scale,
        'time': obs['grid_times'].values,
        'flux': obs['flux'].values,
        'flux_error': obs['flux_error'].values,
        'band_indices': obs['band_indices'].values,
        'time_indices': obs['time_indices'].values,
    }

    astronomical_object.preprocess_data = preprocess_data

    return preprocess_data
