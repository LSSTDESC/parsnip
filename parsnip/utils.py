import avocado
import os
import sys
import sncosmo


def load_dataset(name):
    """Load a dataset using avocado"""
    from .astronomical_object import ParsnipObject
    dataset = avocado.Dataset.load(name, object_class=ParsnipObject)
    return dataset


def load_panstarrs_bandpasses():
    """Download and load the panstarrs bandpasses into sncosmo"""

    path = os.path.join(avocado.settings['data_directory'], 'ps1_bandpasses_mrt.txt')
    url = 'http://iopscience.iop.org/0004-637X/750/2/99/suppdata/apj425122t3_mrt.txt'

    # Download the bandpasses if we don't have them already.
    if not os.path.exists(path):
        print("Downloading PanSTARRS bandpasses...")
        sys.stdout.flush()
        avocado.utils.download_file(url, path)
    
    # Load them into sncosmo
    from astropy.io import ascii

    band_data = ascii.read('../panstarrs/bands/apj425122t3_mrt.txt')

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