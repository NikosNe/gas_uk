#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#!/usr/bin/env python

import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="gas_uk",
    version="0.0.2",
    author="Nikolaos Nezeritis",
    author_email="nikolaosnezeritis@gmail.com",
    description="A package with the modelling process for the gas consumption timeseries somewhere in the UK",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/NikosNe/gas_uk",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)