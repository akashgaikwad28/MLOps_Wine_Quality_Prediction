
from setuptools import setup, find_packages

setup(
    name="Wine_Quality_Prediction",
    version="0.0.1",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    install_requires=[],  # or parse requirements.txt
)
