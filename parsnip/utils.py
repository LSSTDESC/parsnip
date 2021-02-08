import avocado
import sncosmo
import numpy as np
import os


def load_dataset(name, *args, **kwargs):
    """Load a dataset using avocado"""
    from .astronomical_object import ParsnipObject
    dataset = avocado.Dataset.load(name, object_class=ParsnipObject, *args, **kwargs)
    return dataset


def load_panstarrs_bandpasses():
    """Download and load the panstarrs bandpasses into sncosmo"""
    # Figure out where the bandpass data file ended up.
    dirname = os.path.dirname
    package_root_directory = dirname(dirname(os.path.abspath(__file__)))

    path = os.path.join(package_root_directory, 'data', 'apj425122t3_mrt.txt')

    # Load the bandpasses into sncosmo
    from astropy.io import ascii

    band_data = ascii.read(path)

    wave = band_data['Wave'] * 10.

    bands = {
        'ps1::g': band_data['gp1'],
        'ps1::r': band_data['rp1'],
        'ps1::i': band_data['ip1'],
        'ps1::z': band_data['zp1'],
    }

    for band_name, band_data in bands.items():
        band = sncosmo.Bandpass(wave, band_data, name=band_name)
        sncosmo.registry.register(band, force=True)


# Automatically load bandpasses when parsnip is first imported
load_panstarrs_bandpasses()


def frac_to_mag(fractional_difference):
    """Convert a fractional difference to a difference in magnitude.

    Because this transformation is asymmetric for larger fractional changes, we
    take the average of positive and negative differences
    """
    pos_mag = 2.5 * np.log10(1 + fractional_difference)
    neg_mag = 2.5 * np.log10(1 - fractional_difference)
    mag_diff = (pos_mag - neg_mag) / 2.0

    return mag_diff
