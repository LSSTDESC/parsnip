**********************
Reproducing Boone 2021
**********************

Overview
========

The details of the ParSNIP model are documented in Boone 2021. To reproduce all of the
results in that paper, follow the following steps.

Installing ParSNIP
==================

Install the ParSNIP software package following the instructions on the
:doc:`installation` page.

Downloading the data
====================

From the desired working directory, run the following scripts on the command line to
download the PLAsTiCC and PS1 datasets to `./data/` directory.

Download PS1::

    $ lcdata_download_ps1

Download PLAsTiCC (warning, this can take a long time)::

    $ lcdata_download_plasticc

Build a combined PLAsTiCC training set for ParSNIP::

    $ parsnip_build_plasticc_combined
    

Training the ParSNIP model
==========================

Note: Model training is much faster if a GPU is available. By default, ParSNIP will
attempt to use the GPU if there is one and fallback to CPU if not. This can be overriden
by passing e.g. `--device cpu` to the `parsnip_train` script where `cpu` is the desired
PyTorch device.

Train a PS1 model using the full dataset (1 hour)::

    $ parsnip_train \
        ./models/parsnip_ps1.pt \
        ./data/ps1.h5

Train a PS1 model with a held-out validation set (1 hour)::

    $ parsnip_train \
        ./models/parsnip_ps1_validation.pt \
        ./data/ps1.h5 \
        --split_train_test

Train a PLAsTiCC model using the full dataset (1 day)::

    $ parsnip_train \
        ./models/parsnip_plasticc.pt \
        ./data/plasticc_combined.h5

Train a PLAsTiCC model with a held-out validation set (1 day)::

    $ parsnip_train \
        ./models/parsnip_plasticc_validation.pt \
        ./data/plasticc_combined.h5 \
        --split_train_test


Generate predictions
====================

Generate predictions for the PS1 dataset (< 1 min)::

    parsnip_predict ./predictions/parsnip_predictions_ps1.h5 \
        ./models/parsnip_ps1.pt \
        ./data/ps1.h5

Generate predictions for the PS1 dataset with 100-fold augmentation (3 min)::

    parsnip_predict ./predictions/parsnip_predictions_ps1_aug_100.h5 \
        ./models/parsnip_ps1.pt \
        ./data/ps1.h5 \
        --augments 100

Generate predictions for the PLAsTiCC combined training dataset (7 min)::

    parsnip_predict ./predictions/parsnip_predictions_plasticc_combined.h5 \
        ./models/parsnip_plasticc.pt \
        ./data/plasticc_combined.h5

Generate predictions for the PLAsTiCC training set with 100-fold augmentation (4 min)::

    parsnip_predict ./predictions/parsnip_predictions_plasticc_train_aug_100.h5 \
        ./models/parsnip_plasticc.pt \
        ./data/plasticc_train.h5 \
        --augments 100

Generate predictions for the full PLAsTiCC dataset (1 hour)::

    parsnip_predict ./predictions/parsnip_predictions_plasticc_test.h5 \
        ./models/parsnip_plasticc.pt \
        ./data/plasticc_test.h5

Figures and analysis
====================

All of the figures and analysis in Boone 2021 were done with `Jupyter notebooks that are
available on GitHub <https://github.com/LSSTDESC/parsnip/tree/main/notebooks>`_. To rerun
these notebooks, copy the notebooks folder to the working directory and run the
notebooks from within that folder.
