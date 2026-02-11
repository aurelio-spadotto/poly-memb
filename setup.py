from setuptools import setup, find_packages

setup(
    name="polymemb",
    version="0.1.0",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "meshio>=5.3,<6.0",
    ],
)

