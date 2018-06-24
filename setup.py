#!/usr/bin/env python

from setuptools import setup

from pipenv.project import Project
from pipenv.utils import convert_deps_to_pip

pfile = Project(chdir=False).parsed_pipfile
requirements = convert_deps_to_pip(pfile['packages'], r=False)
test_requirements = convert_deps_to_pip(pfile['dev-packages'], r=False)

setup(name="cryptotax",
      version="0.0.1",
      description="meh",
      long_description="meh",
      author="Erik Bjäreholt",
      author_email="erik@bjareho.lt",
      install_requires=requirements + test_requirements,
      packages=set(["cryptotax"]))
