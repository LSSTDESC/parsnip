import numpy as np
import torch


def nmad(x):
    """Calculate the normalize median absolute deviation (NMAD)

    Parameters
    ----------
    x : `~numpy.ndarray`
        Data to calculate the NMAD of

    Returns
    -------
    float
        NMAD of the input
    """
    return 1.4826 * np.median(np.abs(x - np.median(x)))


def frac_to_mag(fractional_difference):
    """Convert a fractional difference to a difference in magnitude.

    Because this transformation is asymmetric for larger fractional changes, we
    take the average of positive and negative differences.

    This supports numpy broadcasting.

    Parameters
    ----------
    fractional_difference : float
        Fractional flux difference

    Returns
    -------
    float
        Difference in magnitudes
    """
    pos_mag = 2.5 * np.log10(1 + fractional_difference)
    neg_mag = 2.5 * np.log10(1 - fractional_difference)
    mag_diff = (pos_mag - neg_mag) / 2.0

    return mag_diff


def parse_device(device):
    """Figure out which PyTorch device to use

    Parameters
    ----------
    device : str
        Requested device

    Returns
    -------
    str
        Device to use
    """
    # Figure out which device to run on.
    if device == 'cpu':
        # Requested CPU.
        use_device = 'cpu'
    elif torch.cuda.is_available():
        # Requested GPU and it is available.
        use_device = device
    else:
        print(f"WARNING: Device '{device}' not available, using 'cpu' instead.")
        use_device = 'cpu'

    return use_device
