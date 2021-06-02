import sncosmo
import numpy as np
import os
from functools import reduce, lru_cache
import lcdata


def parse_ps1(dataset):
    """Parse a PanSTARRS-1 dataset"""
    # Throw out light curves that don't have good redshifts or are otherwise bad.
    dataset = dataset[dataset.meta['unsupervised']]

    # Labels to use for classification
    label_map = {
        'Unknown': 'Unknown',
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
    dataset.meta['type'] = [label_map[i] for i in dataset.meta['type']]

    return dataset


def parse_ztf(dataset):
    """Parse a ZTF dataset"""
    # Throw out light curves that don't have good redshifts.
    dataset = dataset[~dataset.meta['redshift'].isnull()]

    # Throw out observations with zero flux.
    # TODO: Update this.
    raise Exception("Can't handle zero flux observations!")
    # if dataset.objects is not None:
        # new_objects = []
        # for obj in dataset.objects:
            # obs = obj.observations[obj.observations['flux'] != 0.]
            # new_objects.append(type(obj)(obj.meta, obs))
        # dataset = avocado.Dataset.from_objects(dataset.name, new_objects)

    # Clean up labels
    types = [str(i).replace(' ', '').replace('?', '') for i in dataset.meta['type']]
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

    dataset.meta['label'] = [label_map[i] for i in types]

    # Update the label on each object.
    if dataset.objects is not None:
        for obj, label in zip(dataset.objects, dataset.meta['label']):
            obj.meta['label'] = label

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
    dataset = dataset[dataset.meta['label'].isin(valid_classes)]

    return dataset


def parse_plasticc(dataset):
    """Parse a PLAsTiCC dataset"""
    # Throw out light curves that don't look like supernovae
    valid_classes = [
        'SNIa',
        'SNIa-91bg',
        'SNIax',
        'SNII',
        'SNIbc',
        'SLSN-I',
        'TDE',
        'KN',
        'ILOT',
        'CaRT',
        'PISN',
        # 'AGN',
        # 'RRL',
        # 'M-dwarf',
        # 'EB',
        # 'Mira',
        # 'muLens-Single',
        # 'muLens-Binary',
        # 'muLens-String',
    ]

    dataset = dataset[np.isin(dataset.meta['type'], valid_classes)]

    return dataset


def parse_dataset(dataset, path_or_name=None, kind=None, verbose=True):
    """Parse a dataset from the lcdata package.

    We cut out observations that are not relevant for this model (e.g. galactic ones),
    and update the class labels.

    We try to guess the kind of dataset from the filename. If this doesn't work, specify
    the kind explicitly instead.
    """
    if kind is None and path_or_name is not None:
        # Parse the dataset to figure out what we need to do with it.
        parse_name = path_or_name.lower().split('/')[-1]
        if 'ps1' in parse_name or 'panstarrs' in parse_name:
            if verbose:
                print(f"Parsing '{parse_name}' as PanSTARRS dataset ...")
            kind = 'ps1'
        elif 'plasticc' in parse_name:
            if verbose:
                print(f"Parsing '{parse_name}' as PLAsTiCC dataset...")
            kind = 'plasticc'
        elif 'ztf' in parse_name:
            if verbose:
                print(f"Parsing '{parse_name}' as ZTF dataset...")
            kind = 'ztf'
        else:
            if verbose:
                print(f"Unknown dataset type '{parse_name}'. Using default parsing. "
                      "Specify how to parse it in utils.py if necessary.")
            kind = 'default'

    if kind == 'ps1':
        dataset = parse_ps1(dataset)
    elif kind == 'plasticc':
        dataset = parse_plasticc(dataset)
    elif kind == 'ztf':
        dataset = parse_ztf(dataset)
    elif kind == 'default':
        # Don't do anything by default
        pass
    else:
        if verbose:
            print(f"Unknown dataset type '{kind}'. Using default parsing. "
                  "Specify how to parse it in utils.py if necessary.")

    return dataset


def load_dataset(path, kind=None, in_memory=True, verbose=True):
    """Load a dataset using the lcdata package.

    This can be any lcdata HDF5 dataset. We use `parse_dataset` to clean things up for
    ParSNIP by rejecting irrelevant light curves (e.g. galactic ones) and updating class
    labels.

    We try to guess the dataset type from the filename. If this doesn't work, specify
    the filename explicitly instead.
    """
    dataset = lcdata.read_hdf5(path, in_memory=in_memory)
    dataset = parse_dataset(dataset, path, kind=kind, verbose=verbose)

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
    train_mask = np.ones(len(dataset), dtype=bool)
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


@lru_cache(maxsize=None)
def get_band_effective_wavelength(band):
    """Calculate the effective wavelength of a band

    The results of this calculation are cached, and the effective wavelength will only
    be calculated once for each band.

    Parameters
    ----------
    band : str
        Name of a band in the `sncosmo` band registry

    Returns
    -------
    effective_wavelength
        Effective wavelength of the band.
    """
    return sncosmo.get_bandpass(band).wave_eff


def get_bands(dataset):
    """Retrieve a list of bands in a dataset

    Parameters
    ----------
    dataset : `lcdata.Dataset`
        Dataset to retrieve the bands from

    Returns
    -------
    bands
        List of bands in the dataset sorted by effective wavelength
    """
    bands = set()
    for lc in dataset.light_curves:
        bands = bands.union(lc['band'])

    sorted_bands = np.array(sorted(bands, key=get_band_effective_wavelength))

    return sorted_bands
