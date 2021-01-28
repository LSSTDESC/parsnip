from setuptools import setup

setup(
    name='parsnip',
    version='0.1',
    description='Deep Generative Models of Astronomical Transients',
    url='http://github.com/kboone/parsnip',
    author='Kyle Boone',
    author_email='kyboone@uw.edu',
    license='BSD',
    packages=['parsnip'],
    data_files=[('data', ['apj425122t3_mrt.txt'])]
)
