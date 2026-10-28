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
        "scipy",
        "scikit-learn",
        "matplotlib",
        "seaborn",
        "networkx",
        "anndata",
        "scanpy",
        "torch",
        "torch_geometric>=2.0",
        "pygam",
    ],
    python_requires=">=3.10",
)
