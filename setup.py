from setuptools import setup, find_packages

with open("requirements.txt") as f:
    requirements = f.read().splitlines()

setup(
    name="anime_recommender_system",
    version="0.1",
    author="Hrithik",
    packages=find_packages(),
    install_requires=requirements,
)
