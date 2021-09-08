import argparse

from .instruments import calculate_band_mw_extinctions, should_correct_background

default_model = 'plasticc'

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


def update_derived_settings(settings):
    """Update the derived settings for a model

    This calculate the Milky Way extinctions in each band, and determines whether
    background correction should be applied.

    Parameters
    ----------
    settings : dict
        Input settings

    Returns
    -------
    dict
        Updated settings with derived settings calculated
    """

    # Figure out what Milky Way extinction correction to apply for each band.
    settings['band_mw_extinctions'] = calculate_band_mw_extinctions(settings['bands'])

    # Figure out if we want to do background correction for each band.
    settings['band_correct_background'] = should_correct_background(settings['bands'])

    # Flag that the derived settings have been calculated so that we don't redo it when
    # loading a model from disk.
    settings['derived_settings_calculated'] = True

    return settings


def parse_settings(bands, settings={}, ignore_unknown_settings=False):
    """Parse the settings for a ParSNIP model

    Parameters
    ----------
    bands : List[str]
        Bands to use in the encoder model
    settings : dict, optional
        Settings to override, by default {}
    ignore_unknown_settings : bool, optional
        If False (default), raise an KeyError if there are any unknown settings.
        Otherwise, do nothing.

    Returns
    -------
    dict
        Parsed settings dictionary

    Raises
    ------
    KeyError
        If there are unknown keys in the input settings
    """
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
    """Parse a string into a list of integers

    For example, the string "1,2,3,4" will be parsed to [1, 2, 3, 4].

    Parameters
    ----------
    text : str
        String to parse

    Returns
    -------
    List[int]
        Parsed integer list
    """
    result = [int(i) for i in text.split(',')]
    return result


def build_default_argparse(description):
    """Build an argparse object that can handle all of the ParSNIP model settings.

    The resulting parsed namespace can be passed to parse_settings to get a ParSNIP
    settings object.

    Parameters
    ----------
    description : str
        Description for the argument parser

    Returns
    -------
    `~argparse.ArgumentParser`
        Argument parser with the ParSNIP model settings added as arguments
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
