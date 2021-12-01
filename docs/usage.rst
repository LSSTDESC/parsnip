*****
Usage
*****

Overview
========

ParSNIP is a generative model of astronomical transient light curves. It is designed to
work with light curves in `sncosmo` format using the `lcdata` package to handle large
datasets. See the `lcdata` documentation for details on how to download or ingest
different datasets.

Training a model
================

ParSNIP provides a built-in script called `parsnip_train` that can be used to train a
model on an `lcdata` dataset. It takes as input the path that the model will be saved to
along with a list of paths to datasets. For example::


    $ parsnip_train ./model.pt ./dataset_1.h5 ./dataset_2.h5

will train a model named `model.pt` using the datasets `dataset_1.h5` and
`dataset_2.h5`.

Generating predictions
======================

The `parsnip_predict` script can be used to generate predictions given an `lcdata`
dataset and a pretrained ParSNIP model. To run it::

    $ parsnip_predict ./predictions.h5 ./model.h5 ./dataset.h5

will generate predictions to the file named `predictions.h5` using the dataset
`dataset.h5` and the model `model.h5`.

Loading a dataset in Python
===========================

ParSNIP is designed to work with `lcdata` datasets. `lcdata` datasets are guaranteed to
be in a specific format, but they may include instrument-specific quirks, light curves
that are not compatible with ParSNIP, or metadata in unusual formats (e.g. PLAsTiCC
types are random integers). ParSNIP includes tools to clean up datasets from a range of
different surveys and reject invalid light curves. Given an `lcdata` dataset, this can
be done with::

    >>> dataset = parsnip.parse_dataset(raw_dataset, kind='ps1')

Here `kind` specifies the type of dataset, in this case one from PanSTARRS-1. Currently
supported options include:

* ps1
* ztf
* plasticc

A convenience function is also included to read `lcdata` datasets in HDF5 format and
parse them automatically::

    >>> dataset = parsnip.load_dataset('/path/to/data.h5')

This function will attempt to determine the dataset kind from the filename. This can be
overridden with the `kind` keyword as in the previous example.

Loading a model in Python
=========================

Once a model has been trained, ParSNIP has a vast Python API for manipulating it and
using it to generate predictions and plots. To load a model in Python::

    >>> import parsnip
    >>> model = parsnip.load_model('/path/to/model.h5')

There are several built-in models included that can be loaded by specifying their name.
Currently, these are:

* `plasticc` trained on the PLAsTiCC dataset.
* `ps1` trained on the PS1 dataset from Villar et al. 2020.
* `plasticc_photoz` trained on the PLAsTiCC dataset. Uses the photometric redshifts
  instead of the true redshifts.

To load one of these built-in models::

    >>> model = parsnip.load_model('plasticc')

Assuming that you have a light curve in `sncosmo` format, some examples of what can be
done with a model include:

Predict the latent representation of a light curve::

    >>> model.predict(light_curve)
    {
        'object_id': 'PS0909006',
        ...
        's1': 0.19424194,
        's1_error': 0.44743112,
        's2': -0.051611423,
        's2_error': 1.0143535,
        ...
    }

Plot the predicted light curve::

    >>> parsnip.plot_light_curve(light_curve, model)

Plot the predicted spectrum at a given time::

    >>> parsnip.plot_spectrum(light_curve, model, time=53000.)

See the :doc:`reference` page for a list of all of the built-in methods, or the
`notebooks that were used to make figures for Boone et al.
2021 <https://github.com/kboone/parsnip/tree/main/notebooks>`_ for examples.

Classifying light curves
========================

To classify light curves, we first need to predict their representations using a ParSNIP
model. This can be done either with the `parsnip_predict` script described previously or
by operating in memory on an `lcdata` Dataset object::

    >>> predictions = model.predict_dataset(dataset)
    >>> print(predictions)
    object_id    ra      dec     ...       s3        s3_error 
    --------- -------- --------  ... ------------- -----------
    PS0909006 333.9503   1.1848  ...    0.19424233   0.4474311
    PS0909010  37.1182  -4.0789  ...   -0.40881702  0.59658796
    PS0910012  52.4718 -28.0867  ...     -2.142636  0.08176677
    PS0910016  35.3073    -3.91  ...   -0.31671444   0.5740286
          ...      ...      ...  ...           ...         ...

A classifier can be trained on a set of predictions with::

    >>> classifier = parsnip.Classifier()
    >>> classifier.train(predictions)

The classifier can the be used to generate predictions for a new dataset with::

    >>> classifier.predict(new_predictions)
    object_id SLSN  SNII  SNIIn SNIa  SNIbc
    --------- ----- ----- ----- ----- -----
    PS0909006 0.009 0.025 0.031 0.858 0.077
    PS0909010 0.001 0.002 0.017 0.954 0.024
    PS0910016 0.002 0.002 0.017 0.948 0.032
    PSc000001 0.003 0.936 0.038 0.003 0.021
    PSc090022 0.960 0.001 0.037 0.001 0.000
          ...   ...   ...   ...   ...   ...

For more details and examples, see the `classification demo notebook
<https://github.com/kboone/parsnip/blob/main/notebooks/classification.ipynb>`_.
