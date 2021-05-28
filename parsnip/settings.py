import argparse
import extinction
import numpy as np
import sncosmo

default_settings = {
    'model_version': 1,

    'min_wave': 1000.,
    'max_wave': 11000.,
    'spectrum_bins': 300,
    'max_redshift': 4.,
    'band_oversampling': 51,
    'time_window': 300,
    'time_pad': 100,
    'time_sigma': 20.,
    'color_sigma': 0.3,
    'magsys': 'ab',
    'error_floor': 0.01,

    'batch_size': 128,
    'learning_rate': 1e-3,
    'scheduler_factor': 0.5,
    'min_learning_rate': 1e-5,
    'penalty': 1e-3,

    'latent_size': 3,
    'input_redshift': True,
    'encode_block': 'residual',
    'encode_conv_architecture': [40, 80, 120, 160, 200, 200, 200],
    'encode_conv_dilations': [1, 2, 4, 8, 16, 32, 64],
    'encode_fc_architecture': [200],
    'encode_time_architecture': [200],
    'encode_latent_prepool_architecture': [200],
    'encode_latent_postpool_architecture': [200],
    'decode_architecture': [40, 80, 160],

    # Settings that will be filled later.
    'derived_settings_calculated': None,
    'bands': None,
    'band_mw_extinctions': None,
    'band_correct_background': None,
}


band_info = {
    # Information about all of the different bands and how to handle them. We assume
    # that all data from the same telescope should be processed the same way.

    # Band name     Correct     Correct
    #               Background  MWEBV

    # PanSTARRS
    'ps1::g':       (True,      True),
    'ps1::r':       (True,      True),
    'ps1::i':       (True,      True),
    'ps1::z':       (True,      True),

    # PLAsTICC
    'lsstu':        (True,      False),
    'lsstg':        (True,      False),
    'lsstr':        (True,      False),
    'lssti':        (True,      False),
    'lsstz':        (True,      False),
    'lssty':        (True,      False),

    # ZTF
    'ztfr':         (False,     True),
    'ztfg':         (False,     True),
    'ztfi':         (False,     True),

    # SWIFT
    'uvot::u':      (False,     True),
    'uvot::b':      (False,     True),
    'uvot::v':      (False,     True),
    'uvot::uvm2':   (False,     True),
    'uvot::uvw1':   (False,     True),
    'uvot::uvw2':   (False,     True),
}


def calculate_band_mw_extinctions(bands):
    """Calculate the MW extinction corrections to apply for each band.

    Multiply mwebv by these values to get the extinction that should be applied to
    each band for a specific light curve.  For bands that have already been
    corrected, we set this value to 0.
    """
    band_mw_extinctions = []

    for band_name in bands:
        # Check if we should be correcting the extinction for this band.
        try:
            should_correct = band_info[band_name][1]
        except KeyError:
            raise KeyError(f"Can't handle band {band_name}. Add it to band_info in "
                           "settings.py")

        if should_correct:
            band = sncosmo.get_bandpass(band_name)
            band_mw_extinctions.append(extinction.fm07(np.array([band.wave_eff]),
                                                       3.1)[0])
        else:
            band_mw_extinctions.append(0.)

    band_mw_extinctions = np.array(band_mw_extinctions)

    return band_mw_extinctions


def should_correct_background(bands):
    """Figure out if we should correct the background levels for each band."""
    band_correct_background = []

    for band_name in bands:
        # Check if we should be correcting the extinction for this band.
        try:
            should_correct = band_info[band_name][0]
        except KeyError:
            raise KeyError(f"Can't handle band {band_name}. Add it to band_info in "
                           "settings.py")

        band_correct_background.append(should_correct)

    band_correct_background = np.array(band_correct_background)

    return band_correct_background


def update_derived_settings(settings):
    """Update derived settings for a model"""

    # Figure out what Milky Way extinction correction to apply for each band.
    settings['band_mw_extinctions'] = calculate_band_mw_extinctions(settings['bands'])

    # Figure out if we want to do background correction for each band.
    settings['band_correct_background'] = should_correct_background(settings['bands'])

    # Flag that the derived settings have been calculated so that we don't redo it when
    # loading a model from disk.
    settings['derived_settings_calculated'] = True

    return settings


def parse_settings(bands, settings={}, ignore_unknown_settings=False):
    """Parse the settings for a ParSNIP model"""
    if 'derived_settings_calculated' in settings:
        # We are loading a prebuilt-model, don't recalculate everything.
        prebuilt_model = True
    else:
        prebuilt_model = False

    use_settings = default_settings.copy()
    use_settings['bands'] = bands

    for key, value in settings.items():
        if key not in default_settings:
            if ignore_unknown_settings:
                continue
            else:
                raise KeyError(f"Unknown setting '{key}' with value '{value}'.")
        else:
            use_settings[key] = value

    if not prebuilt_model:
        use_settings = update_derived_settings(use_settings)

    return use_settings


def parse_int_list(text):
    result = [int(i) for i in text.split(',')]
    return result


def build_default_argparse(description):
    """Build an argparse object that can handle all of the ParSNIP model settings.

    The resulting parsed namespace can be passed to parse_settings to get a ParSNIP
    settings object.
    """
    parser = argparse.ArgumentParser(description=description)

    for key, value in default_settings.items():
        if value is None:
            # Derived setting, not something that should be specified.
            continue

        if isinstance(value, bool):
            # Handle booleans.
            if value:
                parser.add_argument(f'--no_{key}', action='store_false', dest=key)
            else:
                parser.add_argument(f'--{key}', action='store_true', dest=key)
        elif isinstance(value, list):
            # Handle lists of integers
            parser.add_argument(f'--{key}', type=parse_int_list, default=value)
        else:
            # Handle other object types
            parser.add_argument(f'--{key}', type=type(value), default=value)

    return parser
