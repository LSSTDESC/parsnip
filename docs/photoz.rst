*******************************
Including Photometric Redshifts
*******************************

Overview
========

The base ParSNIP model described in Boone 2021 assumes that the redshift of each
transient is known. In Boone et al. 2022 (in prep.), ParSNIP was extended to handle
datasets that only have photometric redshifts available. ParSNIP uses the photometric
redshift as a prior and predicts the redshift of each transients. Currently ParSNIP only
supports Gaussian photometric redshifts like the ones in the PLAsTiCC dataset, but it is
straightforward to include more complex photometric redshift priors.

The `plasticc_photoz` built-in model was trained on the PLAsTiCC dataset and uses
photometric redshifts instead of true redshifts. It can be loaded with the following
command:

    >>> model = parsnip.load_model('plasticc_photoz')

This model assumes that each transient has metadata with a `hostgal_photoz` key
containing the mean photometric redshift prediction and a `hostgal_photoz_err` key
containing the photometric redshift uncertainty.

Training ParSNIP with photometric redshifts
===========================================

The following steps can be used to train a model that uses photometric redshifts on the
PLAsTiCC dataset and generate predictions for both the training and test datasets. You
should first follow the steps in :doc:`boone2021` to download the PLAsTiCC dataset.

Photometric redshifts are enabled by passing the `--predict_redshift` flag to
`parsnip_train`. Model training can be unstable at early epochs when the redshift is
being predicted, so we recommend using larger batch sizes and starting the training with
a lower learning rate. A batch size of 256 and a learning rate of 5e-4 is stable for
the PLAsTiCC dataset.

Note: Model training is much faster if a GPU is available. By default, ParSNIP will
attempt to use the GPU if there is one and fallback to CPU if not. This can be overriden
by passing e.g. `--device cpu` to the `parsnip_train` script where `cpu` is the desired
PyTorch device.

Train the PLAsTiCC model using the full dataset (1 day)::

    $ parsnip_train \
        ./models/parsnip_plasticc_photoz.pt \
        ./data/plasticc_combined.h5 \
        --batch_size 256 \
        --learning_rate 5e-4 \
        --predict_redshift

Generate predictions for the PLAsTiCC training set with 100-fold augmentation (4 min)::

    parsnip_predict ./predictions/parsnip_predictions_plasticc_photoz_train_aug_100.h5 \
        ./models/parsnip_plasticc_photoz.pt \
        ./data/plasticc_train.h5 \
        --augments 100

Generate predictions for the full PLAsTiCC dataset (1 hour)::

    parsnip_predict ./predictions/parsnip_predictions_plasticc_photoz_test.h5 \
        ./models/parsnip_plasticc_photoz.pt \
        ./data/plasticc_test.h5

By default, ParSNIP uses a spectroscopic redshift prior with a width of 0.01 during
training. This can be adjusted using the `specz_error` flag to `parsnip_train`. For
example, running `parsnip_train ... --specz_error 0.05` will use a prior with a width of
0.05 instead.

Figures and analysis
====================

All of the figures and analysis in Boone et al. 2022 (in prep.) can be reproduced with
`a Jupyter notebook that is available on GitHub
<https://github.com/LSSTDESC/parsnip/tree/main/notebooks/photoz.ipynb>`_. To rerun this
notebook on a newly trained model, copy the notebooks folder to the working directory
and run the notebook from within that folder.
