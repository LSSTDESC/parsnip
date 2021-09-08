*****************
SNCosmo Interface
*****************

Overview
========

ParSNIP provides an SNCosmo interface with an implementation of the `sncosmo.Source`
class. To load the built-in ParSNIP model trained on the PLAsTiCC dataset::

    >>> import parsnip
    >>> source = parsnip.ParsnipSncosmoSource('plasticc')

This source can be used in any SNCosmo models or methods. For example::

    >>> import sncosmo
    >>> model = sncosmo.Model(source=source)

    >>> model.param_names
    ['z', 't0', 'amplitude', 'color', 's1', 's2', 's3']

    >>> data = sncosmo.load_example_data()
    >>> result, fitted_model = sncosmo.fit_lc(
    ...     data, model,
    ...     ['z', 't0', 'amplitude', 's1', 's2', 's3', 'color'],
    ...     bounds={'z': (0.3, 0.7)},
    ... )

Note that ParSNIP is a generative model in that it predicts the full spectral time
series of each transient. When used with the SNCosmo interface, it can operate on light
curves observed in any bands, not just the ones that it was trained on.

Predicting the model parameters with variational inference
==========================================================

The ParSNIP model uses variational inference to predict the posterior distribution over
all of the parameters of the model. An SNCosmo model can be initialized with the result
of this prediction::

    >>> parsnip_model = parsnip.load_model( ... )
    >>> sncosmo_model = parsnip_model.predict_sncosmo(light_curve)
