from setuptools import find_packages, setup

setup(
    name="scifly",
    version="0.0.1",
    author_email="antoine.chevrot@gmail.com",
    description="Take you from raw flight data to ML datasets",
    packages=find_packages(),
    install_requires=[
        "pandas",
        "vincenty",
        "datetime",
        "pyproj",
    ],
)
