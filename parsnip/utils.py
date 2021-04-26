import avocado
import sncosmo
import numpy as np
import os


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


def should_correct_background(bandpass):
    """Check if the background level should be corrected for a given bandpass"""
    try:
        return band_info[bandpass][0]
    except KeyError:
        raise Exception(f"Can't handle bandpass {bandpass}. Add it to band_info in "
                        "utils.py")


def should_correct_mw_extinction(bandpass):
    """Check if Milky Way extinction should be corrected for a given bandpass"""
    try:
        return band_info[bandpass][1]
    except KeyError:
        raise Exception(f"Can't handle bandpass {bandpass}. Add it to band_info in "
                        "utils.py")


def parse_panstarrs(dataset):
    """Parse a PanSTARRS dataset"""
    # Throw out light curves that don't have good redshifts or are otherwise bad.
    dataset = dataset[dataset.metadata['unsupervised']]

    bands = ['ps1::g', 'ps1::r', 'ps1::i', 'ps1::z']
    correct_background = True
    correct_mw_extinction = True

    # Labels to use for classification
    label_map = {
        '-': '-',           # Unknown object
        'FELT': 'FELT',
        'SLSN': 'SLSN',
        'SNII': 'SNII',
        'SNIIb?': 'SNII',
        'SNIIn': 'SNIIn',
        'SNIa': 'SNIa',
        'SNIax': 'SNIax',
        'SNIbc (Ib)': 'SNIbc',
        'SNIbc (Ic)': 'SNIbc',
        'SNIbc (Ic-BL)': 'SNIbc',
        'SNIbn': 'SNIbc',
    }
    dataset.metadata['label'] = [label_map[i] for i in dataset.metadata['type']]

    return dataset, bands, correct_background, correct_mw_extinction


def parse_ztf(dataset):
    """Parse a ZTF dataset"""
    # Throw out light curves that don't have good redshifts.
    dataset = dataset[~dataset.metadata['redshift'].isnull()]

    bands = ['ztfr', 'ztfg', 'ztfi', 'uvot::u', 'uvot::b', 'uvot::v',
             'uvot::uvm2', 'uvot::uvw1', 'uvot::uvw2']
    correct_background = False
    correct_mw_extinction = False

    # Labels to use for classification
    # TODO clean this up
    dataset.metadata['label'] = dataset.metadata['type']

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

    # Labels to use for classification
    label_map = {
        90: 'SNIa',
        67: 'SNIa-91bg',
        52: 'SNIax',
        42: 'SNII',
        62: 'SNIbc',
        95: 'SLSN',
        15: 'TDE',
        64: 'KN',
        992: 'ILOT',
        993: 'CaRT',
        994: 'PISN',
    }
    dataset.metadata['label'] = [label_map[i] for i in dataset.metadata['class']]

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
    # TODO: Get rid of all of this and go back to the original Avocado method of
    # loading datasets.
    # - Need to handle loading multiple datasets in pieces. That could be done by
    # parsing a string, e.g. plasticc_train,plasticc_test[0,100],plasticc_test[5,10]
    # - Need to be able to detect invalid redshifts and throw them out (with a flag?)
    # - Need to be able to detect invalid types of objects and throw them out (with a
    # flag?)
    # - Get rid of ParsnipObject altogether to make things easier. It isn't really
    # necessary with my latest code.

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
    elif dataset_type == 'ztf':
        result = parse_ztf(dataset)
    else:
        raise Exception(f"Unknown dataset {name}. Specify how to handle it in utils.py")

    dataset, bands, correct_background, correct_mw_extinction = result

    if dataset.objects is not None:
        # Flag each object
        for obj in dataset.objects:
            obj.correct_background = correct_background
            obj.correct_mw_extinction = correct_mw_extinction

        # Update the label on each object.
        for obj, label in zip(dataset.objects, dataset.metadata['label']):
            obj.metadata['label'] = label

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
