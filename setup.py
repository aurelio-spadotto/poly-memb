from setuptools import setup, find_packages

# Read dependencies from requirements.txt
with open("requirements.txt") as f:
    requirements = f.read().splitlines()

setup(
    name="polymemb",
    version="0.1.0",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=requirements,
)

