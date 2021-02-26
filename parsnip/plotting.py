from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix
import itertools
import numpy as np


def plot_light_curve(model, obj, count=100, show_model=True, show_bands=True,
                     percentile=68, ax=None, **kwargs):
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6), dpi=100)

    model_times, model_flux, data = model.predict_light_curve(obj, count, **kwargs)

    input_data, compare_data, redshifts, band_indices, amp_scales = data

    band_indices = band_indices.detach().cpu().numpy()
    compare_data = compare_data.detach().cpu().numpy()

    time, flux, fluxerr, weight = compare_data

    max_model = 0.

    for band_idx, band_name in enumerate(model.band_map):
        c = f'C{band_idx}'

        match = band_indices == band_idx
        ax.errorbar(time[match], flux[match], fluxerr[match], fmt='o', c=c,
                    label=band_name, elinewidth=1)

        if band_idx == 0:
            label = 'Model'
        else:
            label = None

        if not show_model:
            # Don't show the model
            band_max_model = np.max(model_flux[band_idx])
        elif count == 0:
            # Single prediction
            ax.plot(model_times, model_flux[band_idx], c=c, label=label)
            band_max_model = np.max(model_flux[band_idx])
        elif show_bands:
            # Multiple predictions, show error bands.
            percentile_offset = (100 - percentile) / 2.
            flux_median = np.median(model_flux[:, band_idx], axis=0)
            flux_min = np.percentile(model_flux[:, band_idx], percentile_offset,
                                     axis=0)
            flux_max = np.percentile(model_flux[:, band_idx],
                                     100 - percentile_offset, axis=0)
            ax.plot(model_times, flux_median, c=c, label=label)
            ax.fill_between(model_times, flux_min, flux_max, color=c, alpha=0.3)
            band_max_model = np.max(flux_median)
        else:
            # Multiple predictions, show raw light curves
            ax.plot(model_times, model_flux[:, band_idx].T, c=c, alpha=0.1)
            band_max_model = np.max(model_flux)

        max_model = max(max_model, band_max_model)

    ax.set_ylim(-0.2 * max_model, 1.2 * max_model)

    ax.legend()
    ax.set_xlabel('Time (days)')
    ax.set_ylabel('Flux')


def plot_spectrum(model, obj, time, count=100, show_bands=True, percentile=68,
                  ax=None, c=None, label=None):
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6), dpi=100)

    model_wave = model.model_wave
    model_spectra = model.predict_spectrum(obj, time, count)

    if count == 0:
        # Single prediction
        ax.plot(model_wave, model_spectra, c=c, label=label)
    elif show_bands:
        # Multiple predictions, show error bands.
        percentile_offset = (100 - percentile) / 2.
        flux_median = np.median(model_spectra, axis=0)
        flux_min = np.percentile(model_spectra, percentile_offset, axis=0)
        flux_max = np.percentile(model_spectra, 100 - percentile_offset, axis=0)
        ax.plot(model_wave, flux_median, c=c, label=label)
        ax.fill_between(model_wave, flux_min, flux_max, color=c, alpha=0.3)
    else:
        # Multiple predictions, show raw light curves
        ax.plot(model_wave, model_spectra.T, c=c, alpha=0.1)

    ax.set_xlabel('Wavelength ($\\AA$)')
    ax.set_ylabel('Flux')


def plot_confusion_matrix(predictions, classifications):
    """Plot a confusion matrix

    Adapted from example that used to be at
    http://scikit-learn.org/stable/modules/generated/sklearn.metrics.confusion_matrix.html
    """
    if len(classifications.columns) == 2 and classifications.columns[0] == 'Other':
        # Single class classification. All labels other than the target one are grouped
        # as "Other".
        target_label = classifications.columns[1]
        match_labels = predictions['label'] == target_label
        labels = match_labels.replace({False: 'Other', True: target_label})
    else:
        labels = predictions['label']

    class_names = classifications.columns

    plt.figure(figsize=(5, 4), dpi=100)
    cm = confusion_matrix(labels, classifications.idxmax(axis=1),
                          labels=class_names, normalize='true')

    im = plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues, vmin=0, vmax=1)
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=60, ha='right')
    plt.yticks(tick_marks, class_names)

    fmt = '.2f'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True class')
    plt.xlabel('Predicted class')
    plt.tight_layout()

    # Make a colorbar that is lined up with the plot
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    ax = plt.gca()
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="4%", pad=0.25)
    plt.colorbar(im, cax=cax, label='Fraction of objects')
