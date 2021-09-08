***************
Reference / API
***************

.. currentmodule:: parsnip


Models
======

*Loading/saving a model*

.. autosummary::
   :toctree: api

   ParsnipModel
   load_model
   ParsnipModel.save
   ParsnipModel.to

*Interacting with a dataset*

.. autosummary::
   :toctree: api

   ParsnipModel.preprocess
   ParsnipModel.augment_light_curves
   ParsnipModel.get_data_loader
   ParsnipModel.fit
   ParsnipModel.score

*Generating model predictions*

.. autosummary::
   :toctree: api

   ParsnipModel.predict
   ParsnipModel.predict_dataset
   ParsnipModel.predict_dataset_augmented
   ParsnipModel.predict_light_curve
   ParsnipModel.predict_spectrum
   ParsnipModel.predict_sncosmo

*Individual parts of the model*

.. autosummary::
   :toctree: api

   ParsnipModel.forward
   ParsnipModel.encode
   ParsnipModel.decode
   ParsnipModel.decode_spectra
   ParsnipModel.loss_function


Datasets
========

*Loading datasets*

.. autosummary::
   :toctree: api

   load_dataset
   load_datasets
   parse_dataset

*Parsers for specific instruments*

.. autosummary::
   :toctree: api

   parse_plasticc
   parse_ps1
   parse_ztf

*Tools for manipulating datasets*

.. autosummary::
   :toctree: api

   split_train_test
   get_bands


Plotting
========

.. autosummary::
   :toctree: api

   plot_light_curve
   plot_representation
   plot_spectrum
   plot_spectra
   plot_sne_space
   plot_confusion_matrix
   get_band_plot_color
   get_band_plot_marker


Classification
==============

.. autosummary::
   :toctree: api

   Classifier
   extract_top_classifications
   weighted_multi_logloss


SNCosmo Interface
=================

.. autosummary::
   :toctree: api

   ParsnipSncosmoSource
   ParsnipModel.predict_sncosmo


Custom Neural Network Layers
============================

.. autosummary::
   :toctree: api

   ResidualBlock
   Conv1dBlock
   GlobalMaxPoolingTime


Settings
========

.. autosummary::
   :toctree: api

   update_derived_settings
   parse_settings
   parse_int_list
   build_default_argparse


Light curve utilities
=====================

.. autosummary::
   :toctree: api

   preprocess_light_curve
   time_to_grid
   grid_to_time
   get_band_effective_wavelength
   calculate_band_mw_extinctions
   should_correct_background


General utilities
=================

.. autosummary::
   :toctree: api

   nmad
   frac_to_mag
   parse_device
