from setuptools import setup, find_packages

setup(
    name="plastinet",
    version="v0.1.8",
    description="Plastinet is Python package for spatial transcriptomics.",
    author="Izabella Zamora",
    author_email="zamora@broadinstitute.org",
    url="https://github.com/izabellaleahz/plastinet",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "pandas",
        "scikit-learn",
        "matplotlib",
        "scanpy",
        "squidpy",
        "gudhi",
        "cmcrameri",
    ],
    python_requires=">=3.10",
)
