try:
    from setuptools import setup, find_packages
except ImportError:
    from distutils.core import setup, find_packages

setup(name='tensornex',
      version='0.1.0',
      description='A collection of customized tensor methods.',
      url='https://github.com/cyrillustan/tensornex',
      license='MIT',
      packages=find_packages(exclude=['doc']),
      install_requires=['numpy', 'tensorly', 'scikit-learn'])
