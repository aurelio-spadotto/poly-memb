from setuptools import setup, find_packages

# Read the contents of your README file
from pathlib import Path
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()

setup(
    name="poly-memb",
    version="0.1.0",
    author="Aurelio Spadotto",
    author_email="aurelio-edoardo.spadotto@umontpellier.fr",
    description="A 2D suite to implement polytopal methods ",
    long_description=long_description,
    long_description_content_type="text/markdown",  # Ensure this matches your README format
    url="https://github.com/yourusername/my-library",  # Replace with your GitHub repo URL
    packages=find_packages(),  # Automatically find sub-packages
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",  # Update license as needed
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.0',
    install_requires=[
        "numpy>=1.20.3",
        "scipy>=1.7.1",
        "sympy>=1.9.0"
        "matplotlib>=3.4.3",
        "meshio>=5.1.1"
    ],
    extras_require={
        "dev": [
            "sphinx>=5.3.0",
        ]
    },
    include_package_data=True,  # Include files specified in MANIFEST.in
    entry_points={
        # Add CLI entry points if needed
        "console_scripts": [
            "my-library=my_library.cli:main",
        ],
    },
)
