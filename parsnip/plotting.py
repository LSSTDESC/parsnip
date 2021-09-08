from matplotlib import pyplot as plt
from matplotlib.gridspec import GridSpec
from sklearn.metrics import confusion_matrix
import itertools
import numpy as np

from .light_curve import preprocess_light_curve
from .classifier import extract_top_classifications
from .instruments import get_band_plot_color, get_band_plot_marker


def _get_reference_time(light_curve):
    if 'reference_time' in light_curve.meta:
        # Reference time calculated from running through the full ParSNIP model.
        return light_curve.meta['reference_time']
    elif 'parsnip_reference_time' in light_curve.meta:
        # Initial estimate of the reference time.
        return light_curve.meta['parsnip_reference_time']
    else:
        # No estimate of the reference time. Just show the light curve as is.
        return 0.


def plot_light_curve(light_curve, model=None, count=100, show_uncertainty_bands=True,
                     show_missing_bandpasses=False, percentile=68, normalize_flux=False,
                     sncosmo_model=None, sncosmo_label='SNCosmo Model', ax=None):
    """Plot a light curve

    Parameters
    ----------
    light_curve : `~astropy.table.Table`
        Light curve to plot
    model : `~ParsnipModel`, optional
        ParSNIP model to show, by default None
    count : int, optional
        Number of samples from the ParSNIP model, by default 100
    show_uncertainty_bands : bool, optional
        If True (default), show uncertainty bands. Otherwise, show individual draws.
    show_missing_bandpasses : bool, optional
        Whether to show model predictions for bandpasses where there is no data, by
        default False
    percentile : int, optional
        Percentile for the uncertainty bands, by default 68
    normalize_flux : bool, optional
        Whether to normalize the flux, by default False
    sncosmo_model : `~sncosmo.Model`, optional
        SNCosmo model to show, by default None
    sncosmo_label : str, optional
        Legend label for the SNCosmo model, by default 'SNCosmo Model'
    ax : axis, optional
        Matplotlib axis to use for the plot, by default one will be created
    """
    if model is not None:
        light_curve = preprocess_light_curve(light_curve, model.settings)

    if ax is None:
        fig, ax = plt.subplots(figsize=(5, 4), constrained_layout=True)

    used_bandpasses = []

    if normalize_flux:
        flux_scale = 1. / light_curve.meta['parsnip_scale']
    else:
        flux_scale = 1.

    reference_time = _get_reference_time(light_curve)

    band_groups = light_curve.group_by('band').groups
    for band_name, band_data in zip(band_groups.keys['band'], band_groups):
        if len(band_data) == 0:
            continue

        c = get_band_plot_color(band_name)
        marker = get_band_plot_marker(band_name)

        band_time = band_data['time']
        band_flux = band_data['flux'] * flux_scale
        band_fluxerr = band_data['fluxerr'] * flux_scale
        band_time = band_time - reference_time

        ax.errorbar(band_time, band_flux, band_fluxerr, fmt='o', c=c, label=band_name,
                    elinewidth=1, marker=marker)

        used_bandpasses.append(band_name)

    if model is not None:
        max_model = 0.
        label_model = True

        model_times, model_flux, model_result = model.predict_light_curve(
            light_curve, sample=True, count=count
        )

        model_times = model_times - reference_time
        model_flux = model_flux * flux_scale

        for band_idx, band_name in enumerate(model.settings['bands']):
            if band_name not in used_bandpasses and not show_missing_bandpasses:
                continue

            c = get_band_plot_color(band_name)
            marker = get_band_plot_marker(band_name)

            if label_model:
                label = 'ParSNIP Model'
                label_model = False
            else:
                label = None

            if count == 0:
                # Single prediction
                ax.plot(model_times, model_flux[band_idx], c=c, label=label)
                band_max_model = np.max(model_flux[band_idx])
            elif show_uncertainty_bands:
                # Multiple predictions, show error bands.
                percentile_offset = (100 - percentile) / 2.
                flux_median = np.median(model_flux[:, band_idx], axis=0)
                flux_min = np.percentile(model_flux[:, band_idx], percentile_offset,
                                         axis=0)
                flux_max = np.percentile(model_flux[:, band_idx],
                                         100 - percentile_offset, axis=0)
                ax.plot(model_times, flux_median, c=c, label=label)
                ax.fill_between(model_times, flux_min,
                                flux_max, color=c, alpha=0.3)
                band_max_model = np.max(flux_median)
            else:
                # Multiple predictions, show raw light curves
                ax.plot(model_times, model_flux[:, band_idx].T, c=c, alpha=0.1)
                band_max_model = np.max(model_flux)

            max_model = max(max_model, band_max_model)

        ax.set_ylim(-0.2 * max_model, 1.2 * max_model)

    if sncosmo_model is not None:
        model_times = np.arange(sncosmo_model.mintime(), sncosmo_model.maxtime(), 0.5)

        for band_idx, band_name in enumerate(model.settings['bands']):
            if band_name not in used_bandpasses and not show_missing_bandpasses:
                continue

            try:
                flux = flux_scale * sncosmo_model.bandflux(
                    band_name, model_times, zp=25., zpsys='ab'
                )
            except ValueError:
                # Outside of wavelength range
                continue

            c = get_band_plot_color(band_name)
            if band_idx == 0:
                label = sncosmo_label
            else:
                label = None

            ax.plot(model_times - reference_time, flux, c=c, ls='--', label=label)

    ax.legend()

    if reference_time != 0.:
        ax.set_xlabel(f'Relative Time (days + {reference_time:.2f})')
    else:
        ax.set_xlabel('Time (days)')

    if normalize_flux:
        ax.set_ylabel('Normalized Flux')
    else:
        ax.set_ylabel('Flux ($ZP_{AB}$=25)')


def normalize_spectrum_flux(wave, flux, min_wave=5500., max_wave=6500.):
    """Normalize the flux of a spectrum

    The flux will be normalized so that it averages to 1 in a given window.

    Parameters
    ----------
    wave : `~numpy.ndarray`
        Wavelengths
    flux : `~numpy.ndarray`
        Flux values
    min_wave : float, optional
        Minimum wavelength to consider for normalization, by default 5500.
    max_wave : float, optional
        Maximum wavelength to consider for normalization, by default 6500.

    Returns
    -------
    `~numpy.ndarray`
        Normalized flux
    """
    cut = (wave > min_wave) & (wave < max_wave)
    scale = np.mean(flux[..., cut], axis=-1)
    return (flux.T / scale).T


def plot_spectrum(light_curve, model, time, count=100, show_uncertainty_bands=True,
                  percentile=68, ax=None, c=None, label=None, offset=None,
                  normalize_flux=False, normalize_min_wave=5500.,
                  normalize_max_wave=6500., spectrum_label=None,
                  spectrum_label_wave=7500., spectrum_label_offset=0.2, flux_scale=1.):
    """Plot the spectrum of a light curve predicted by a ParSNIP model

    Parameters
    ----------
    light_curve : `~astropy.table.Table`
        Light curve
    model : `~ParsnipModel`
        Model to use for the prediction
    time : float
        Time to predict the spectrum at
    count : int, optional
        Number of spectra to sample, by default 100
    show_uncertainty_bands : bool, optional
        Whether to show uncertainty bands, by default True
    percentile : int, optional
        Percentile for the uncertainty bands, by default 68
    ax : axis, optional
        Matplotlib axis to use, by default None
    c : str, optional
        Color for the plot, by default None
    label : str, optional
        Label for the plot, by default None
    offset : float, optional
        Constant offset to add to the flux for plotting, by default None
    normalize_flux : bool, optional
        Whether to normalize the flux, by default False
    normalize_min_wave : float, optional
        Minimum wavelength of the normalization window, by default 5500.
    normalize_max_wave : float, optional
        Maximum wavelength of the normalization window, by default 6500.
    spectrum_label : str, optional
        Label to plot near the spectrum, by default None
    spectrum_label_wave : float, optional
        Wavelength to plot the spectrum label at, by default 7500.
    spectrum_label_offset : float, optional
        Y offset for the spectrum label, by default 0.2
    flux_scale : float, optional
        Scale to multiply the flux by, by default 1.
    """
    light_curve = preprocess_light_curve(light_curve, model.settings)

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6), dpi=100)

    model_wave = model.model_wave
    model_spectra = model.predict_spectrum(light_curve, time, sample=True, count=count)

    if normalize_flux:
        model_spectra = normalize_spectrum_flux(
            model_wave, model_spectra, normalize_min_wave, normalize_max_wave
        )

    model_spectra *= flux_scale

    if offset is not None:
        model_spectra += offset

    if count == 0:
        # Single prediction
        ax.plot(model_wave, model_spectra, c=c, label=label)
    elif show_uncertainty_bands:
        # Multiple predictions, show error bands.
        percentile_offset = (100 - percentile) / 2.
        flux_median = np.median(model_spectra, axis=0)
        flux_min = np.percentile(model_spectra, percentile_offset, axis=0)
        flux_max = np.percentile(
            model_spectra, 100 - percentile_offset, axis=0)
        ax.plot(model_wave, flux_median, c=c, label=label)
        ax.fill_between(model_wave, flux_min, flux_max, color=c, alpha=0.3)
    else:
        # Multiple predictions, show raw light curves
        ax.plot(model_wave, model_spectra.T, c=c, alpha=0.1)

    if spectrum_label is not None:
        # Show a label above the spectrum.
        wave_idx = np.searchsorted(model.model_wave, spectrum_label_wave)
        label_height = spectrum_label_offset + np.mean(model_spectra[..., wave_idx])
        ax.text(spectrum_label_wave, label_height, spectrum_label)

    ax.set_xlabel('Wavelength ($\\AA$)')
    if normalize_flux:
        ax.set_ylabel('Normalized Flux')
    else:
        ax.set_ylabel('Flux')


def plot_spectra(light_curve, model, times=[0., 10., 20., 30.], flux_scale=1.,
                 ax=None, sncosmo_model=None, sncosmo_label='SNCosmo Model',
                 spectrum_label_offset=0.2):
    """Plot the spectral time series of a light curve predicted by a ParSNIP model

    Parameters
    ----------
    light_curve : `~astropy.table.Table`
        Light curve
    model : `~ParsnipModel`
        Model to use for the predictions
    times : list, optional
        Times to predict the spectra at, by default [0., 10., 20., 30.]
    flux_scale : float, optional
        Scale to multiple the flux by, by default 1.
    ax : axis, optional
        Matplotlib axis, by default None
    sncosmo_model : `~sncosmo.Model`, optional
        SNCosmo model to overplot, by default None
    sncosmo_label : str, optional
        Label for the SNCosmo model, by default 'SNCosmo Model'
    spectrum_label_offset : float, optional
        Offset of the time labels for each spectrum, by default 0.2
    """
    light_curve = preprocess_light_curve(light_curve, model.settings)

    wave = model.model_wave
    redshift = light_curve.meta['redshift']
    scale = flux_scale / light_curve.meta['parsnip_scale']

    if ax is None:
        fig, ax = plt.subplots(figsize=(5, 4), constrained_layout=True)

    reference_time = _get_reference_time(light_curve)

    for plot_idx, time in enumerate(times):
        plot_offset = len(times) - plot_idx - 1

        plot_time = time + reference_time

        if plot_idx == 0:
            use_label = 'ParSNIP Model'
            use_sncosmo_label = sncosmo_label
        else:
            use_label = None
            use_sncosmo_label = None

        plot_spectrum(light_curve, model, plot_time, ax=ax, c='C2',
                      flux_scale=scale, offset=plot_offset, label=use_label,
                      spectrum_label=f'{time:+.1f} days',
                      spectrum_label_offset=spectrum_label_offset)

        if sncosmo_model is not None:
            sncosmo_flux = (
                sncosmo_model._flux(plot_time, wave * (1 + redshift))[0]
                * 10**(0.4 * 45)
                * (1 + redshift)
            )
            ax.plot(wave, scale * sncosmo_flux + plot_offset, c='k', alpha=0.3,
                    label=use_sncosmo_label)

    ax.set_ylabel('Normalized Flux + Offset')
    ax.legend()


def plot_sne_space(light_curve, model, name, min_wave=10000., max_wave=0., time_diff=0.,
                   min_time=-10000., max_time=100000., source=None, kernel=5,
                   flux_scale=0.5, label_wave=9000., label_offset=0.2, figsize=(5, 6)):
    """Compare a ParSNIP spectrum prediction to a real spectrum from sne.space

    Parameters
    ----------
    light_curve : `~astropy.table.Table`
        Light curve
    model : `~ParsnipModel`
        ParSNIP Model to use for the prediction
    name : str
        Name of the light curve on sne.space
    min_wave : float, optional
        Ignore any spectra that don't have data bluer than this wavelength, by default
        10000.
    max_wave : float, optional
        Ignore any spectra that don't have data redder than this wavelength, by default
        0.
    time_diff : float, optional
        Minimum time between spectra, by default 0.
    min_time : float, optional
        Ignore any spectra before this time, by default -10000.
    max_time : float, optional
        Ignore any spectra after this time, by default 100000.
    source : str, optional
        Ignore any spectra not from this source, by default None
    kernel : int, optional
        Smooth the spectra by a median filter kernel of this size, by default 5
    flux_scale : float, optional
        Scale the flux by this amount, by default 0.5
    label_wave : float, optional
        Show labels with the times of each spectrum at this wavelength, by default 9000.
    label_offset : float, optional
        Y offset to use for the labels, by default 0.2
    figsize : tuple, optional
        Figure size, by default (5, 6)
    """
    import json
    import urllib
    from scipy.signal import medfilt

    light_curve = preprocess_light_curve(light_curve, model.settings)

    redshift = light_curve.meta['redshift']
    reference_time = _get_reference_time(light_curve)

    fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)

    url = f'https://api.sne.space/{name}/spectra/time+data+instrument+telescope+source'
    with urllib.request.urlopen(url) as request:
        data = json.loads(request.read().decode())

    spectra = data[name]['spectra']

    plot_idx = 0
    last_time = 1e9

    for spec in spectra[::-1]:
        # Go in reverse order so that we can plot the spectra with the
        # first one on top.
        spec_time, spec_data, telescope, _, spec_source = spec

        spec_time = float(spec_time)
        spec_data = np.array(spec_data, dtype=float)

        spec_wave = spec_data[:, 0] / (1 + redshift)
        spec_flux = spec_data[:, 1]
        spec_flux = medfilt(spec_flux, kernel)

        if spec_wave[0] > min_wave or spec_wave[-1] < max_wave:
            # print("skipping wave")
            continue

        if last_time - spec_time < time_diff:
            continue

        if (spec_time - reference_time < min_time
                or spec_time - reference_time > max_time):
            continue

        if source is not None and spec_source != source:
            continue

        last_time = spec_time

        normalize_min_wave = max([5500., spec_wave[0]])
        normalize_max_wave = min([6500., spec_wave[-1]])

        plot_offset_scale = 1.
        plot_offset = (plot_idx * plot_offset_scale)

        plot_spectrum(
            light_curve,
            model,
            spec_time,
            normalize_flux=True,
            normalize_min_wave=normalize_min_wave,
            normalize_max_wave=normalize_max_wave,
            flux_scale=flux_scale,
            ax=ax,
            offset=plot_offset,
            c='C2'
        )

        spec_flux = flux_scale * normalize_spectrum_flux(
            spec_wave, spec_flux, normalize_min_wave, normalize_max_wave
        )
        ax.plot(spec_wave, spec_flux + plot_offset, c='k')

        plt.text(label_wave, plot_offset + label_offset,
                 f'${spec_time - reference_time:.1f}$ days', ha='right')

        plot_idx += 1

    plt.legend(['ParSNIP Model', 'Observed Spectra'])
    plt.title("")
    plt.xlabel('Rest-Frame Wavelength ($\\AA$)')
    plt.ylabel('Normalized Flux + Offset')
    plt.xlim(1500., 10500.)


def plot_confusion_matrix(predictions, classifications, figsize=(5, 4), title=None,
                          verbose=True):
    """Plot a confusion matrix

    Adapted from example that used to be at
    http://scikit-learn.org/stable/modules/generated/sklearn.metrics.confusion_matrix.html

    Parameters
    ----------
    predictions : `~astropy.table.Table`
        Predictions from `~ParsnipModel.predict_dataset`
    classifications : `~astropy.table.Table`
        Classifications from a `~Classifier`
    figsize : tuple, optional
        Figure size, by default (5, 4)
    title : str, optional
        Figure title, by default None
    verbose : bool, optional
        Whether to print additional statistics, by default True
    """
    true_types = np.char.decode(predictions['type'])
    predicted_types = extract_top_classifications(classifications)

    if len(classifications.columns) == 3 and classifications.colnames[2] == 'Other':
        # Single class classification. All labels other than the target one are grouped
        # as "Other".
        single_type = classifications.colnames[1]
        true_types[true_types != single_type] = 'Other'

    type_names = classifications.colnames[1:]

    plt.figure(figsize=figsize, constrained_layout=True)
    cm = confusion_matrix(true_types, predicted_types, labels=type_names,
                          normalize='true')

    im = plt.imshow(cm, interpolation='nearest',
                    cmap=plt.cm.Blues, vmin=0, vmax=1)
    tick_marks = np.arange(len(type_names))
    plt.xticks(tick_marks, type_names, rotation=60, ha='right')
    plt.yticks(tick_marks, type_names)

    fmt = '.2f'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True Type')
    plt.xlabel('Predicted Type')
    if title is not None:
        plt.title(title)

    # Make a colorbar that is lined up with the plot
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    ax = plt.gca()
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="4%", pad=0.25)
    plt.colorbar(im, cax=cax, label='Fraction of objects')

    if verbose:
        # Print out stats.
        print("Macro averaged completeness (Villar et al. 2020): "
              f"{np.diag(cm).mean():.4f}")
        print(f"Fraction correct: {np.mean(true_types == predicted_types):.4f}")


def plot_representation(predictions, plot_labels, mask=None, idx1=1, idx2=2, idx3=None,
                        max_count=1000, show_legend=True, legend_ncol=1, marker=None,
                        markersize=5, ax=None):
    """Plot the representation of a ParSNIP model

    Parameters
    ----------
    predictions : `~astropy.table.Table`
        Predictions for a dataset from `~ParsnipModel.predict_dataset`
    plot_labels : List[str]
        Labels for each of the classes
    mask : `~np.array`, optional
        Mask to apply to the predictions, by default None
    idx1 : int, optional
        Intrinsic latent variable to plot on the x axis, by default 1
    idx2 : int, optional
        Intrinsic latent variable to plot on the y axis, by default 2
    idx3 : int, optional
        If specified, show a three paneled plot with this latent variable in the extra
        two panels plotted against the other ones
    max_count : int, optional
        Maximum number of light curves to show of each type, by default 1000
    show_legend : bool, optional
        Whether to show the legend, by default True
    legend_ncol : int, optional
        Number of columns to use in the legend, by default 1
    marker : str, optional
        Matplotlib marker to use, by default None
    markersize : int, optional
        Matplotlib marker size, by default 5
    ax : axis, optional
        Matplotlib axis, by default None
    """
    color_map = {
        'SNIa': 'C0',
        'SNIax': 'C9',
        'SNIa-91bg': 'lightgreen',

        'SLSN': 'C2',
        'SLSN-I': 'C2',
        'SNII': 'C1',
        'SNIIn': 'C3',
        'SNIbc': 'C4',

        'KN': 'C5',

        'CaRT': 'C3',
        'ILOT': 'C6',
        'PISN': 'C8',
        'TDE': 'C7',

        'FELT': 'C5',
        'Peculiar': 'C5',
    }

    if idx3 is not None:
        if ax is not None:
            raise Exception("Can't make 3D plot with prespecified axis.")

        fig = plt.figure(figsize=(8, 8), constrained_layout=True)

        gs = GridSpec(2, 2, figure=fig)

        ax12 = fig.add_subplot(gs[1, 0])
        ax13 = fig.add_subplot(gs[0, 0], sharex=ax12)
        ax32 = fig.add_subplot(gs[1, 1], sharey=ax12)
        legend_ax = fig.add_subplot(gs[0, 1])
        legend_ax.axis('off')

        plot_vals = [
            (idx1, idx2, ax12),
            (idx1, idx3, ax13),
            (idx3, idx2, ax32),
        ]
    else:
        if ax is None:
            fig, ax = plt.subplots(figsize=(6, 6), constrained_layout=True)

        plot_vals = [
            (idx1, idx2, ax)
        ]

    for xidx, yidx, ax in plot_vals:
        if mask is not None:
            cut_predictions = predictions[~mask]
            ax.scatter(cut_predictions[f's{xidx}'], cut_predictions[f's{yidx}'],
                       c='k', s=3, alpha=0.1, label='Unknown')
            valid_predictions = predictions[mask]
        else:
            valid_predictions = predictions

        for type_name in plot_labels:
            type_predictions = valid_predictions[valid_predictions['type'] ==
                                                 type_name]

            color = color_map[type_name]

            markers, caps, bars = ax.errorbar(
                type_predictions[f's{xidx}'][:max_count],
                type_predictions[f's{yidx}'][:max_count],
                xerr=type_predictions[f's{xidx}_error'][:max_count],
                yerr=type_predictions[f's{yidx}_error'][:max_count],
                label=type_name,
                fmt='o',
                marker=marker,
                markersize=markersize,
                c=color,
            )

            [bar.set_alpha(0.3) for bar in bars]

    if idx3 is not None:
        ax12.set_xlabel(f'$s_{idx1}$')
        ax12.set_ylabel(f'$s_{idx2}$')
        ax13.set_ylabel(f'$s_{idx3}$')
        ax13.tick_params(labelbottom=False)
        ax32.set_xlabel(f'$s_{idx3}$')
        ax32.tick_params(labelleft=False)

        if show_legend:
            handles, labels = ax12.get_legend_handles_labels()
            legend_ax.legend(handles=handles, labels=labels, loc='center',
                             ncol=legend_ncol)
    else:
        ax.set_xlabel(f'$s_{idx1}$')
        ax.set_ylabel(f'$s_{idx2}$')

        if show_legend:
            ax.legend(ncol=legend_ncol)
