import avocado
import sncosmo
import numpy as np
import os
from functools import reduce


def parse_panstarrs(dataset):
    """Parse a PanSTARRS dataset"""
    # Throw out light curves that don't have good redshifts or are otherwise bad.
    dataset = dataset[dataset.metadata['unsupervised']]

    # Labels to use for classification
    label_map = {
        '-': 'Unknown',            # Unknown object
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

    # Update the label on each object.
    if dataset.objects is not None:
        for obj, label in zip(dataset.objects, dataset.metadata['label']):
            obj.metadata['label'] = label

    return dataset


def parse_ztf(dataset):
    """Parse a ZTF dataset"""
    # Throw out light curves that don't have good redshifts.
    dataset = dataset[~dataset.metadata['redshift'].isnull()]

    # Throw out observations with zero flux.
    if dataset.objects is not None:
        new_objects = []
        for obj in dataset.objects:
            obs = obj.observations[obj.observations['flux'] != 0.]
            new_objects.append(type(obj)(obj.metadata, obs))
        dataset = avocado.Dataset.from_objects(dataset.name, new_objects)

    # Clean up labels
    types = [str(i).replace(' ', '').replace('?', '') for i in dataset.metadata['type']]
    label_map = {
        'AGN': 'Galaxy',
        'Bogus': 'Bad',
        'CLAGN': 'Galaxy',
        'CV': 'Star',
        'CVCandidate': 'Star',
        'Duplicate': 'Bad',
        'Galaxy': 'Galaxy',
        'Gap': 'Peculiar',
        'GapI': 'Peculiar',
        'GapI-Ca-rich': 'Peculiar',
        'IIP': 'SNII',
        'ILRT': 'Peculiar',
        'LBV': 'Star',
        'LINER': 'Galaxy',
        'LRN': 'Star',
        'NLS1': 'Galaxy',
        'None': 'Bad',
        'Nova': 'Star',
        'Q': 'Galaxy',
        'QSO': 'Galaxy',
        'SLSN-I': 'SLSN',
        'SLSN-I.5': 'SLSN',
        'SLSN-II': 'SLSN',
        'SLSN-R': 'SLSN',
        'SN': 'Unknown',
        'SNII': 'SNII',
        'SNII-pec': 'SNII',
        'SNIIL': 'SNII',
        'SNIIP': 'SNII',
        'SNIIb': 'SNII',
        'SNIIn': 'SNII',
        'SNIa': 'SNIa',
        'SNIa-91T': 'SNIa',
        'SNIa-91T-like': 'SNIa',
        'SNIa-91bg': 'SNIa',
        'SNIa-99aa': 'SNIa',
        'SNIa-CSM': 'SNIa',
        'SNIa-norm': 'SNIa',
        'SNIa-pec': 'SNIa',
        'SNIa00cx-like': 'SNIa',
        'SNIa02cx-like': 'SNIa',
        'SNIa02ic-like': 'SNIa',
        'SNIa91T': 'SNIa',
        'SNIa91T-like': 'SNIa',
        'SNIa91bg-like': 'SNIa',
        'SNIapec': 'SNIa',
        'SNIax': 'SNIa',
        'SNIb': 'SNIbc',
        'SNIb/c': 'SNIbc',
        'SNIbn': 'SNIbc',
        'SNIbpec': 'SNIbc',
        'SNIc': 'SNIbc',
        'SNIc-BL': 'SNIbc',
        'SNIc-broad': 'SNIbc',
        'Star': 'Star',
        'TDE': 'TDE',
        'Var': 'Star',
        'asteroid': 'Asteroid',
        'blazar': 'Galaxy',
        'bogus': 'Bad',
        'duplicate': 'Bad',
        'galaxy': 'Galaxy',
        'nan': 'Unknown',
        'nova': 'Star',
        'old': 'Bad',
        'rock': 'Asteroid',
        'star': 'Star',
        'stellar': 'Star',
        'unclassified': 'Unknown',
        'unk': 'Unknown',
        'unknown': 'Unknown',
        'varstar': 'Star',
    }

    dataset.metadata['label'] = [label_map[i] for i in types]

    # Update the label on each object.
    if dataset.objects is not None:
        for obj, label in zip(dataset.objects, dataset.metadata['label']):
            obj.metadata['label'] = label

    # Drop light curves that aren't supernova-like
    valid_classes = [
        'SNIa',
        'SNII',
        'Unknown',
        # 'Galaxy',
        'SNIbc',
        'SLSN',
        # 'Star',
        'TDE',
        # 'Bad',
        'Peculiar',
    ]
    dataset = dataset[dataset.metadata['label'].isin(valid_classes)]

    return dataset


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

    # Update the label on each object.
    if dataset.objects is not None:
        for obj, label in zip(dataset.objects, dataset.metadata['label']):
            obj.metadata['label'] = label

    return dataset


def load_dataset(name, *args, verbose=True, **kwargs):
    """Load a dataset using avocado.

    This can be any avocado dataset, but we do some additional preprocessing here to
    clean things up for parsnip.  We also cut out observations that are not relevant
    for this model (e.g.  galactic ones).  We do this in a naive way by looking at
    the first word in the dataset name and applying the corresponding preprocessing
    function.
    """
    # The name can contain chunk information in the format name,num_chunks,chunk.  For
    # example, ps1,1,100 will load chunk #1 of 100 from the ps1 dataset.
    dataset_info = name.split(',')
    if len(dataset_info) == 1:
        name = dataset_info[0]
    elif len(dataset_info) == 3:
        name, chunk, num_chunks = dataset_info
        try:
            chunk = int(chunk)
            num_chunks = int(num_chunks)
        except ValueError:
            raise Exception(f"Invalid dataset string '{name}'")
        kwargs['num_chunks'] = num_chunks
        kwargs['chunk'] = chunk
    else:
        raise Exception(f"Invalid dataset string '{name}'")

    # Load the dataset as is.
    dataset = avocado.Dataset.load(name, *args, **kwargs)

    # Parse the dataset to figure out what we need to do with it.
    parse_name = name.lower()
    if 'ps1' in parse_name or 'panstarrs' in parse_name:
        if verbose:
            print(f"Parsing PanSTARRS dataset '{name}'...")
        dataset = parse_panstarrs(dataset)
    elif 'plasticc' in parse_name:
        if verbose:
            print(f"Parsing PLAsTiCC dataset '{name}'...")
        dataset = parse_plasticc(dataset)
    elif 'ztf' in parse_name:
        if verbose:
            print(f"Parsing ZTF dataset '{name}'...")
        dataset = parse_ztf(dataset)
    else:
        if verbose:
            print(f"Unknown dataset type '{name}'. Using default parsing. Specify "
                  "how to parse it in utils.py if necessary.")

    return dataset


def load_datasets(dataset_names, verbose=True):
    """Load a list of datasets and merge them"""
    # Load the dataset(s).
    datasets = []
    for dataset_name in dataset_names:
        datasets.append(load_dataset(dataset_name, verbose=verbose))

    # Add all of the datasets together
    dataset = reduce(lambda i, j: i+j, datasets)

    return dataset


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
    # TODO: Download this data if it isn't on disk.
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
