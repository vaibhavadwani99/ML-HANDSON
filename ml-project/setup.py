from setuptools import find_packages, setup

setup(
    name = 'housing_price_prediction',
    version = 0.2,
    packages = find_packages(where="src"),
    package_dir = {"":"src"}
)