from setuptools import setup, find_packages

setup(
    name='influencemax-optimization',
    version='0.1',
    packages=find_packages(), 
    install_requires=["torch","tqdm","lightning","jax","flax"],
)