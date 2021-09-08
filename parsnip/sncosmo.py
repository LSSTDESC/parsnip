from scipy.interpolate import interp1d
import numpy as np
import os
import sncosmo
import torch

import parsnip

from .light_curve import SIDEREAL_SCALE


class ParsnipSncosmoSource(sncosmo.Source):
    """SNCosmo interface for a ParSNIP model

    Parameters
    ----------
    model : `~ParsnipModel` or str, optional
        ParSNIP model to use, or path to a model on disk.
    """
    def __init__(self, model=None):
        if not isinstance(model, parsnip.ParsnipModel):
            model = parsnip.load_model(model)

        self._model = model

        model_name = os.path.splitext(os.path.basename(model.path))[0]
        self.name = f'parsnip_{model_name}'
        self._param_names = (
            ['amplitude', 'color']
            + [f's{i+1}' for i in range(self._model.settings['latent_size'])]
        )
        self.param_names_latex = (
            ['A', 'c'] + [f's_{i+1}' for i in
                          range(self._model.settings['latent_size'])]
        )
        self.version = 1

        self._parameters = np.zeros(len(self._param_names))
        self._parameters[0] = 1.

    def _flux(self, phase, wave):
        # Generate predictions at the given phase.
        encoding = (torch.FloatTensor(self._parameters[2:])[None, :]
                    .to(self._model.device))
        phase = phase * SIDEREAL_SCALE
        phase = torch.FloatTensor(phase)[None, :].to(self._model.device)
        color = torch.FloatTensor([self._parameters[1]]).to(self._model.device)
        amplitude = (torch.FloatTensor([self._parameters[0]]).to(self._model.device))

        model_spectra = self._model.decode_spectra(encoding, phase, color, amplitude)
        model_spectra = model_spectra.detach().cpu().numpy()[0]

        flux = interp1d(self._model.model_wave, model_spectra.T)(wave)

        return flux

    def minphase(self):
        return (-self._model.settings['time_window'] // 2
                - self._model.settings['time_pad'])

    def maxphase(self):
        return (self._model.settings['time_window'] // 2
                + self._model.settings['time_pad'])

    def minwave(self):
        return self._model.settings['min_wave']

    def maxwave(self):
        return self._model.settings['max_wave']
