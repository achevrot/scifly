from setuptools import setup

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
        "tensorflow",
    ],
)
