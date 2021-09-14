import lcdata
import numpy as np
import scipy.stats

from astropy.stats import biweight_location

SIDEREAL_SCALE = 86400. / 86164.0905


def _determine_time_grid(light_curve):
    """Determine the time grid that will be used for a light curve

    ParSNIP evaluates all light curves on a grid internally for the encoder. This
    function determines where to line up that grid.

    Parameters
    ----------
    light_curve : `~astropy.table.Table`
        Light curve

    Returns
    -------
    float
        Reference time for the time grid
    """
    time = light_curve['time']
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
    s2n = light_curve['flux'] / light_curve['fluxerr']
    s2n_mask = np.argsort(s2n)[-5:]

    # If we have very few observations, only keep the ones above signal-to-noise of
    # 5 if possible. Sometimes we only have a single point on the rise so far, so
    # we don't want to include a bunch of bad observations in our determination of
    # the time.
    s2n_mask_2 = s2n[s2n_mask] > 5.
    if np.any(s2n_mask_2):
        cut_times = shift_time[s2n_mask][s2n_mask_2]
    else:
        # No observations with signal-to-noise above 5. Just use whatever we
        # have...
        cut_times = shift_time[s2n_mask]

    max_time = np.round(np.median(cut_times))

    # Convert back to a reference time in the original units. This reference time
    # corresponds to the reference of the grid in sidereal time.
    reference_time = ((max_time + sidereal_offset) / SIDEREAL_SCALE)
    return reference_time


def time_to_grid(time, reference_time):
    """Convert a time in the original units to one on the internal ParSNIP grid

    Parameters
    ----------
    time : float
        Real time to convert
    reference_time : float
        Reference time for the grid

    Returns
    -------
    float
        Time on the internal grid
    """
    return (time - reference_time) * SIDEREAL_SCALE


def grid_to_time(grid_time, reference_time):
    """Convert a time on the internal grid to a time in the original units

    Parameters
    ----------
    grid_time : float
        Time on the internal grid
    reference_time : float
        Reference time for the grid

    Returns
    -------
    float
        Time in original units
    """
    return grid_time / SIDEREAL_SCALE + reference_time


def preprocess_light_curve(light_curve, settings, raise_on_invalid=True):
    """Preprocess a light curve for the ParSNIP model

    Parameters
    ----------
    light_curve : `~astropy.Table`
        Raw light curve
    settings : dict
        ParSNIP model settings
    raise_on_invalid : bool
        Whether to raise a ValueError for invalid light curves. If False, None is
        returned instead. By default, True.

    Returns
    -------
    `~astropy.Table`
        Preprocessed light curve

    Raises
    ------
    ValueError
        For any invalid light curves that cannot be handled by ParSNIP if
        raise_on_invalid is True. The error message will describe why the light curve is
        invalid.
    """
    if light_curve.meta.get('parsnip_preprocessed', False):
        # Already preprocessed
        return light_curve

    # Parse the light curve with lcdata to ensure that all of the columns/metadata have
    # standard names.
    try:
        light_curve = lcdata.parse_light_curve(light_curve)
    except ValueError as e:
        if raise_on_invalid:
            raise
        else:
            lcdata.utils.warn_first_time("invalid_lc_format",
                                         f"Failed to parse light curve: {e}")
            return None

    # We require that the light curve has a valid redshift.
    if not np.isfinite(light_curve.meta['redshift']):
        message = "No redshift available for light curve."
        if raise_on_invalid:
            raise ValueError(message)
        else:
            lcdata.utils.warn_first_time("missing_redshift", message)
            return None

    # Align the observations to a grid in sidereal time.
    reference_time = _determine_time_grid(light_curve)

    # Build a preprocessed light curve object.
    new_lc = light_curve.copy()

    # Map each band to its corresponding index.
    band_map = {j: i for i, j in enumerate(settings['bands'])}
    new_lc['band_index'] = [band_map.get(i, -1) for i in new_lc['band']]

    # Cut out any observations that are outside of the window that we are
    # considering.
    grid_times = time_to_grid(new_lc['time'], reference_time)
    time_indices = np.round(grid_times).astype(int) + settings['time_window'] // 2
    time_mask = (
        (time_indices >= -settings['time_pad'])
        & (time_indices < settings['time_window'] + settings['time_pad'])
    )
    new_lc['grid_time'] = grid_times
    new_lc['time_index'] = time_indices

    # Correct background levels for bands that need it.
    for band_idx, do_correction in enumerate(settings['band_correct_background']):
        if not do_correction:
            continue

        band_mask = new_lc['band_index'] == band_idx
        # Find observations outside of our window.
        outside_obs = new_lc[~time_mask & band_mask]
        if len(outside_obs) == 0:
            # No outside observations, don't update the background level.
            continue

        # Estimate the background level and subtract it.
        background = biweight_location(outside_obs['flux'])
        new_lc['flux'][band_mask] -= background

    # Cut out observations that are in unused bands or outside of the time window.
    band_mask = new_lc['band_index'] != -1
    new_lc = new_lc[band_mask & time_mask]

    if len(new_lc) == 0:
        # No valid observations for this light curve.
        message = (f"Light curve has no usable observations! Valid bands are "
                   f"{settings['bands']}.")
        if raise_on_invalid:
            raise ValueError(message)
        else:
            lcdata.utils.warn_first_time("unusable_observations", message)
            return None

    # Correct for Milky Way extinction if desired.
    band_extinctions = (
        settings['band_mw_extinctions'] * new_lc.meta.get('mwebv', 0.)
    )
    extinction_scales = 10**(0.4 * band_extinctions[new_lc['band_index']])
    new_lc['flux'] *= extinction_scales
    new_lc['fluxerr'] *= extinction_scales

    # Scale the light curve so that its peak has an amplitude of roughly 1. We use
    # the brightest observation with signal-to-noise above 5 if there is one, or
    # simply the brightest observation otherwise.
    s2n = new_lc['flux'] / new_lc['fluxerr']
    s2n_mask = s2n > 5.
    if np.any(s2n_mask):
        scale = np.max(new_lc['flux'][s2n_mask])
    else:
        scale = np.max(new_lc['flux'])

    new_lc.meta['parsnip_reference_time'] = reference_time
    new_lc.meta['parsnip_scale'] = scale
    new_lc.meta['parsnip_preprocessed'] = True

    return new_lc
