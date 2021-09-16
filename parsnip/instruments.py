from functools import reduce, lru_cache
import extinction
import hashlib
import lcdata
import numpy as np
import sncosmo

"""This file contains instrument specific definitions."""


band_info = {
    # Information about all of the different bands and how to handle them. We assume
    # that all data from the same telescope should be processed the same way.

    # Band name     Correct     Correct     Plot color      Plot marker
    #               Background  MWEBV
    # PanSTARRS
    'ps1::g':       (True,      True,       'C0',           'o'),
    'ps1::r':       (True,      True,       'C2',           '^'),
    'ps1::i':       (True,      True,       'C1',           'v',),
    'ps1::z':       (True,      True,       'C3',           '<'),

    # PLAsTICC
    'lsstu':        (True,      False,      'C6',           'o'),
    'lsstg':        (True,      False,      'C4',           'v'),
    'lsstr':        (True,      False,      'C0',           '^'),
    'lssti':        (True,      False,      'C2',           '<'),
    'lsstz':        (True,      False,      'C3',           '>'),
    'lssty':        (True,      False,      'goldenrod',    's'),

    # ZTF
    'ztfr':         (False,     True,       'C0',           'o'),
    'ztfg':         (False,     True,       'C2',           '^'),
    'ztfi':         (False,     True,       'C1',           'v'),

    # SWIFT
    'uvot::u':      (False,     True,       'C6',           '<'),
    'uvot::b':      (False,     True,       'C3',           '>'),
    'uvot::v':      (False,     True,       'goldenrod',    's'),
    'uvot::uvm2':   (False,     True,       'C5',           'p'),
    'uvot::uvw1':   (False,     True,       'C7',           'P'),
    'uvot::uvw2':   (False,     True,       'C8',           '*'),
}


def calculate_band_mw_extinctions(bands):
    """Calculate the Milky Way extinction corrections for a set of bands

    Multiply mwebv by these values to get the extinction that should be applied to
    each band for a specific light curve. For bands that have already been corrected, we
    set this value to 0.

    Parameters
    ----------
    bands : List[str]
        Bands to calculate the extinction for

    Returns
    -------
    `~numpy.ndarray`
        Milky Way extinction in each band

    Raises
    ------
    KeyError
        If any bands are not available in band_info in instruments.py
    """
    band_mw_extinctions = []

    for band_name in bands:
        # Check if we should be correcting the extinction for this band.
        try:
            should_correct = band_info[band_name][1]
        except KeyError:
            raise KeyError(f"Can't handle band {band_name}. Add it to band_info in "
                           "instruments.py")

        if should_correct:
            band = sncosmo.get_bandpass(band_name)
            band_mw_extinctions.append(extinction.fm07(np.array([band.wave_eff]),
                                                       3.1)[0])
        else:
            band_mw_extinctions.append(0.)

    band_mw_extinctions = np.array(band_mw_extinctions)

    return band_mw_extinctions


def should_correct_background(bands):
    """Determine if we should correct the background levels for a set of bands

    Parameters
    ----------
    bands : List[str]
        Bands to lookup

    Returns
    -------
    `~numpy.ndarray`
        Boolean for each band indicating if it needs background correction

    Raises
    ------
    KeyError
        If any bands are not available in band_info in instruments.py
    """
    band_correct_background = []

    for band_name in bands:
        # Check if we should be correcting the extinction for this band.
        try:
            should_correct = band_info[band_name][0]
        except KeyError:
            raise KeyError(f"Can't handle band {band_name}. Add it to band_info in "
                           "instruments.py")

        band_correct_background.append(should_correct)

    band_correct_background = np.array(band_correct_background)

    return band_correct_background


def get_band_plot_color(band):
    """Return the plot color for a given band.

    If the band does not yet have a color assigned to it, then a random color
    will be assigned (in a systematic way).

    Parameters
    ----------
    band : str
        Name of the band to use.

    Returns
    -------
    str
        Matplotlib color to use when plotting the band
    """
    if band in band_info:
        return band_info[band][2]

    # Systematic random colors. We use the hash of the band name.
    # Note: hash() uses a random offset in python 3 so it isn't consistent
    # between runs!
    hasher = hashlib.md5()
    hasher.update(band.encode("utf8"))
    hex_color = "#%s" % hasher.hexdigest()[-6:]

    return hex_color


def get_band_plot_marker(band):
    """Return the plot marker for a given band.

    If the band does not yet have a marker assigned to it, then we use the
    default circle.

    Parameters
    ----------
    band : str
        Name of the band to use.

    Returns
    -------
    str
        Matplotlib marker to use when plotting the band
    """
    if band in band_info:
        return band_info[band][3]
    else:
        return 'o'


def parse_ps1(dataset):
    """Parse a PanSTARRS-1 dataset

    Parameters
    ----------
    dataset : `~lcdata.Dataset`
        PanSTARRS-1 dataset to parse

    Returns
    -------
    `~lcdata.Dataset`
        Parsed dataset
    """
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
    """Parse a ZTF dataset

    Parameters
    ----------
    dataset : `~lcdata.Dataset`
        ZTF dataset to parse

    Returns
    -------
    `~lcdata.Dataset`
        Parsed dataset
    """
    lcs = []
    for lc in dataset.light_curves:
        # Throw out light curves that don't have valid redshifts.
        if np.isnan(lc.meta['redshift']):
            continue

        # Some ZTF datasets replace lower limits with a flux of zero. This is bad. Throw
        # out all of those observations.
        lc = lc[(lc['flux'] != 0.) & (lc['fluxerr'] != 0.)]
        if len(lc) == 0:
            continue

        lcs.append(lc)
    dataset = lcdata.from_light_curves(lcs)

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
        'None': 'Unknown',
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

    dataset.meta['original_type'] = dataset.meta['type']
    dataset.meta['type'] = [label_map[i] for i in types]

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
    dataset = dataset[np.isin(dataset.meta['type'], valid_classes)]

    return dataset


def parse_plasticc(dataset):
    """Parse a PLAsTiCC dataset

    Parameters
    ----------
    dataset : `~lcdata.Dataset`
        PLAsTiCC dataset to parse

    Returns
    -------
    `lcdata.Dataset`
        Parsed dataset
    """
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

    We cut out observations that are not relevant for the ParSNIP model (e.g. galactic
    ones), and update the class labels.

    We try to guess the kind of dataset from the filename. If this doesn't work, specify
    the kind explicitly instead.

    Parameters
    ----------
    dataset : `~lcdata.Dataset`
        Dataset to parse
    path_or_name : str, optional
        Name of the dataset, or path to it, by default None
    kind : str, optional
        Kind of dataset, by default None
    verbose : bool, optional
        If true, print parsing information, by default True

    Returns
    -------
    `~lcdata.Dataset`
        Parsed dataset
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
                      "Specify how to parse it in instruments.py if necessary.")
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
                  "Specify how to parse it in instruments.py if necessary.")

    return dataset


def load_dataset(path, kind=None, in_memory=True, verbose=True):
    """Load a dataset using the lcdata package.

    This can be any lcdata HDF5 dataset. We use `~parse_dataset` to clean things up for
    ParSNIP by rejecting irrelevant light curves (e.g. galactic ones) and updating class
    labels.

    We try to guess the dataset type from the filename. If this doesn't work, specify
    the filename explicitly instead.

    Parameters
    ----------
    path : str
        Path to the dataset on disk
    kind : str, optional
        Kind of dataset, by default we will attempt to determine it from the filename
    in_memory : bool, optional
        If False, don't load the light curves into memory, and only load the metadata.
        See `lcdata.Dataset` for details.
    verbose : bool, optional
        If True, print parsing information, by default True

    Returns
    -------
    `~lcdata.Dataset`
        Loaded dataset
    """
    dataset = lcdata.read_hdf5(path, in_memory=in_memory)
    dataset = parse_dataset(dataset, path, kind=kind, verbose=verbose)

    return dataset


def load_datasets(dataset_paths, verbose=True):
    """Load a list of datasets and merge them

    Parameters
    ----------
    dataset_paths : List[str]
        Paths to each dataset to load
    verbose : bool, optional
        If True, print parsing information, by default True

    Returns
    -------
    `~lcdata.Dataset`
        Loaded dataset
    """
    # Load the dataset(s).
    datasets = []
    for dataset_name in dataset_paths:
        datasets.append(load_dataset(dataset_name, verbose=verbose))

    # Add all of the datasets together
    dataset = reduce(lambda i, j: i+j, datasets)

    return dataset


def split_train_test(dataset):
    """Split a dataset into training and testing parts.

    We train on 90%, and test on 10%. We use a fixed algorithm to split the train and
    test so that we don't have to keep track of what we did.

    Parameters
    ----------
    dataset : `~lcdata.Dataset`
        Dataset to split

    Returns
    -------
    `~lcdata.Dataset`
        Training dataset
    `~lcdata.Dataset`
        Test dataset
    """
    # Keep part of the dataset for validation
    train_mask = np.ones(len(dataset), dtype=bool)
    train_mask[::10] = False
    test_mask = ~train_mask

    train_dataset = dataset[train_mask]
    test_dataset = dataset[test_mask]

    return train_dataset, test_dataset


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
    float
        Effective wavelength of the band.
    """
    return sncosmo.get_bandpass(band).wave_eff


def get_bands(dataset):
    """Retrieve a list of bands in a dataset

    Parameters
    ----------
    dataset : `~lcdata.Dataset`
        Dataset to retrieve the bands from

    Returns
    -------
    List[str]
        List of bands in the dataset sorted by effective wavelength
    """
    bands = set()
    for lc in dataset.light_curves:
        bands = bands.union(lc['band'])

    sorted_bands = np.array(sorted(bands, key=get_band_effective_wavelength))

    return sorted_bands
