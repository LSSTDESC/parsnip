import avocado
import sncosmo
import numpy as np
import argparse
import os


def parse_panstarrs(dataset):
    """Parse a PanSTARRS dataset"""
    # Throw out light curves that don't have good redshifts or are otherwise bad.
    dataset = dataset[dataset.metadata['unsupervised']]

    bands = ['ps1::g', 'ps1::r', 'ps1::i', 'ps1::z']
    correct_background = True
    correct_mw_extinction = True

    return dataset, bands, correct_background, correct_mw_extinction


def parse_plasticc(dataset):
    """Parse a PLAsTiCC dataset"""
    # Throw out light curves that don't look like supernovae
    valid_classes = [
        90,     # Ia
        67,     # Ia-91bg
        52,     # Iax
        42,     # II
        62,     # Ibc
        95,     # SLSN
        15,     # TDE
        64,     # KN
        # 88,    # AGN
        # 92,    # RR Lyrae
        # 65,    # M-dwarf stellar flare
        # 16,    # Eclipsing binary
        # 53,    # Mira variable
        # 6,     # Microlens
        # 991,   # binary microlens
        992,    # Intermediate luminosity optical transient
        993,    # Calcium rich transient
        994,    # Pair instability SN
    ]

    dataset = dataset[dataset.metadata['class'].isin(valid_classes)]

    bands = ['lsstu', 'lsstg', 'lsstr', 'lssti', 'lsstz', 'lssty']
    correct_background = True
    correct_mw_extinction = False

    return dataset, bands, correct_background, correct_mw_extinction


def load_dataset(name, *args, **kwargs):
    """Load a dataset using avocado.

    This can be any avocado dataset, but we do some additional preprocessing here to
    get it in the format that is needed for parsnip. Specifically, we need to specify
    the bands and their ordering for each dataset, and we need to specify whether to
    apply background corrections and MW extinction corrections. We also cut out
    observations that are not relevant for this model (e.g. galactic ones). We do
    this in a naive way by looking at the first word in the dataset name and applying
    the corresponding preprocessing function.

    We also allow for the special dataset "plasticc_combo" that selects a subset of
    the PLAsTiCC data from the training and validation sets.
    """
    from .astronomical_object import ParsnipObject

    if name == 'plasticc_combo':
        # Load a subset of the PLAsTiCC dataset for training. We can't fit the whole
        # dataset in memory at once, so we only use part of it for the unsupervised
        # training.
        dataset = (
            # Training set
            avocado.load('plasticc_train', object_class=ParsnipObject)

            # DDF set
            + avocado.load('plasticc_test', object_class=ParsnipObject, chunk=0,
                           num_chunks=100)

            # WFD set, load 10% of it
            + avocado.load('plasticc_test', object_class=ParsnipObject, chunk=5,
                           num_chunks=10)
        )
    else:
        # Load the dataset as is.
        dataset = avocado.Dataset.load(name, object_class=ParsnipObject, *args,
                                       **kwargs)

    # Parse the dataset to figure out what we need to do with it.
    dataset_type = name.split('_')[0]

    if dataset_type in ['ps1', 'panstarrs']:
        result = parse_panstarrs(dataset)
    elif dataset_type == 'plasticc':
        result = parse_plasticc(dataset)
    else:
        raise Exception(f"Unknown dataset {name}. Specify how to handle it in utils.py")

    dataset, bands, correct_background, correct_mw_extinction = result

    if dataset.objects is not None:
        # Flag each object
        for obj in dataset.objects:
            obj.correct_background = correct_background
            obj.correct_mw_extinction = correct_mw_extinction

    return dataset, bands


def split_train_test(dataset):
    """Split a dataset into training and testing parts.

    We train on 90%, and test on 10%.
    """
    # Keep part of the dataset for validation
    train_mask = np.ones(len(dataset.objects), dtype=bool)
    train_mask[::10] = False
    test_mask = ~train_mask

    train_dataset = dataset[train_mask]
    test_dataset = dataset[test_mask]

    return train_dataset, test_dataset


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
