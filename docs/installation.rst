************
Installation
************

ParSNIP requires Python 3.6+ and depends on the following Python packages:

- `astropy <http://www.astropy.org>`_
- `extinction <https://github.com/kbarbary/extinction>`_
- `lcdata <https://github.com/kboone/lcdata>`_
- `lightgbm <https://lightgbm.readthedocs.io/en/latest/>`_
- `matplotlib <https://matplotlib.org>`_
- `numpy <http://www.numpy.org>`_
- `scipy <https://scipy.org>`_
- `PyTorch <https://pytorch.org>`_
- `scikit-learn <https://scikit-learn.org/>`_
- `tqdm <https://github.com/tqdm/tqdm>`_

Install using pip (recommended)
===============================

ParSNIP is available on PyPI. To install the latest release::

    pip install astro-parsnip


Install development version
===========================

The ParSNIP source code can be found on `github <https://github.com/kboone/parsnip>`_.

To install it::

    git clone git://github.com/kboone/parsnip
    cd parsnip
    pip install -e .
