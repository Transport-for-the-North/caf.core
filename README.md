![Transport for the North Logo](https://github.com/Transport-for-the-North/caf.toolkit/blob/main/docs/TFN_Landscape_Colour_CMYK.png)

<h1 align="center">CAF template</h1>

[//]: # (Use the below and update URLs to add more badges when deploying)

[//]: # (<a href="https://pypi.org/project/caf.toolkit/"><img alt="Supported Python versions" src="https://img.shields.io/pypi/pyversions/caf.package.svg?style=flat-square"></a>)

[//]: # (<a href="https://pypi.org/project/caf.toolkit/"><img alt="Latest release" src="https://img.shields.io/github/release/transport-for-the-north/caf.package.svg?style=flat-square&maxAge=86400"></a>)

[//]: # (<a href="https://app.codecov.io/gh/Transport-for-the-North/caf.toolkit"><img alt="Coverage" src="https://img.shields.io/codecov/c/github/transport-for-the-north/caf.package.svg?branch=master&style=flat-square&logo=CodeCov"></a>)

[//]: # (<a href="https://github.com/Transport-for-the-North/caf.toolkit/actions?query=event%3Apush"><img alt="Testing Badge" src="https://img.shields.io/github/actions/workflow/status/transport-for-the-north/caf.toolkit/tests.yml?style=flat-square&logo=GitHub&label=Tests"></a>)


<p align="center">
<a href="https://www.gnu.org/licenses/gpl-3.0.en.html"><img alt="License: GNU GPL v3.0" src="https://img.shields.io/badge/license-GPLv3-blueviolet.svg?style=flat-square"></a>
<a href="https://github.com/psf/black"><img alt="code style: black" src="https://img.shields.io/badge/code%20format-black-000000.svg"></a>
</p>


A template repo for caf modules.

For an example implementation of this template please see [caf.toolkit](https://github.com/Transport-for-the-North/caf.toolkit)

## Usage
Follow the instruction below to use this template to make a new caf package based on the
standard caf structure. Briefly:

- Update the package names in all places that {package_name} is used.
- Run and install versioneer

### Places to change package name
In short - anywhere there is something named {package_name} 

- `src/caf/{package_name}`
- `RELEASE.md` - 1 change
  - Update the URL in the 'releases' link
- `pyproject.toml` - 2 changes
  - Under `project` update name 
  - Update the URLs under `project.urls`
  - Under `tool.setuptools.package-data` update caf.{package_name}, e.g., `caf.toolkit = py.typed`
  - Under `tool.mypy` update packages
  - Under `tool.mypy.overrides` update module path
  - Under `tool.pytest.ini_options` update `--cov` argument


### VersioningIt
[VersioningIt](https://github.com/jwodder/versioningit)
is an automatic versioning tool for GitHub based projects. It provides 
consistent and predictable naming based on the number of commits since the last 
user defined version. This way it can be used to find old versions of code, even 
if it's just a random commit in your repo!
The version strings are user configurable, but currently configured to generate
a unique version string for every commit. See the `versioningit` section at the 
bottom of `pyporject.toml` for full details.

Thanks to all the files in this repo, VersioningIt is fully set up and ready 
to use! It will work with deployed packages, and those using editable installs.
Editable installs are useful for local testing during development, and can be 
done by running the following in the root of a package:

`pip install -e .`

VersioningIt is based off of git Tags, which you can set on GitHub. Version 
tags should start with a 'v' and contain three numbers (following the 
[Semantic Versioning](https://semver.org/) convention) e.g., `v0.1.0` for an 
initial version that isn't ready for a full `v1.0.0` release.


## What does this template provide?
This template sets up a lot of CI/CD (Continuous Integration / Continuous 
Deployment) tools to help manage, update, release, and test a new python 
package. Here is a list of what this sets up for you:

- Automatic and easy to use code linting / analysis which works on your machine via tox, which provides:
  - MyPy type checking
  - Pylint syntax checking
  - PyDocStyle documentation checking
  - Test running via pytest
  - Can can be run with `tox` from the root of this repo
- Setup for [Black](https://github.com/psf/black) code formatter
  - This can be run with `black src` or `black tests` from the root of this repo
- GitHub actions which run on all pull requests and pushes to master
  - These run the above tox and black checks and will warn you where code deviates from the standards
- Automatic code versioning via Git Tags
- Lays out the package in a consistent format to fit the `CAF` structure.

## Structure

### docs
All docs go in here. There is only one special folder here named `sphinx` - and this is where the sphinx documentation would go once implemented.
Remove the txt file in the `docs/sphinx` folder. 

Any folders can be added alongside sphinx for package needs.

### src/caf
All code goes in here. Some files already exist:
- `__init__.py` - DO NOT REMOVE. Informs the package builder that this package is part of the greater `caf` package. It makes the familiy of package easier to find.
  Technically, it defines `caf` as a namespace package i.e., it doesn't contain code, but other packages.
- `/{package_name}/py.typed` - this tells python and PyPI that your package is typed, and it should look for type hints in the code.


### tests
All tests go here.
Tests should be written in pytest and should follow the same structure as the src package (minus the src/caf/{package_name}).
See the [pytest](https://docs.pytest.org/en/7.2.x/) documentation for full detail, or [caf.toolkit](https://github.com/Transport-for-the-North/caf.toolkit) for an example.


### files
There's a few files stored in the root of the pacakge which are standard setup files. They are listed and detail below:

- `Contributing.rst` - Standard CAF contribution guidelines. Details on coding standards etc.
- `pyproject.toml` - A file of settings and metadata for the package. This file details how to build the package and defines common linter tool setup.
- `RELEASE.md` - A standard file which should be used to track change notes between package versions.
- `requirements.txt` - Details the packages and their versions that this package depends on. It's a 
  list of the python packages which must be installed for this package to work. Update this file as your package gains dependencies. 
- `requirements_dev.txt` - Details the packages and their versions that this package depends on during 
  testing and linting. These are extra dependencies on top of the `requirements.txt` ones. This is used 
  by package tools to ensure your tests pass when you say they should! This file likely doesn't need changing very often.
- `setup.cfg` - Mostly deprecated in favour of `pyproject.toml`. Used here to be compatible with [versioningit](https://github.com/jwodder/versioningit)
- `tox.ini` - A configuration file for running all tests, linters, and code analysers. Can be run 
  by running `tox` in a terminal. This file is also used by GitHub actions to automatically run
  the same checks when a pull request is made.


## Future work
- Implement Sphinx documentation building setup
