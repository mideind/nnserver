"""
Reynir: Natural language processing for Icelandic

Copyright (C) 2019 Miðeind ehf.

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.
This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see http://www.gnu.org/licenses/.

"""

from setuptools import find_packages, setup

setup(
    name="nnserver",
    version="1.0.0",
    description="Neural Network Middleware Transcoder",
    author="Miðeind",
    author_email="haukur.barri@gmail.com",
    url="http://github.com/mideind/nnserver",
    license="GPLv3+",
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: GNU General Public License v3 or later (GPLv3+)",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    packages=find_packages(),
    scripts=["nnserver/bin/nnserver"],
    package_data={"nnserver": ["resources/*"]},
    install_requires=[
        'gevent<=1.4',
        "flask",
        "tensorflow<2",
        "tensor2tensor",
        "reynir==1.3.1",
        "tokenizer==1.0.8",
    ],
)
