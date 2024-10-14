![Transport for the North Logo](https://github.com/Transport-for-the-North/caf.base/blob/main/docs/TFN_Landscape_Colour_CMYK.png)

<h1 align="center">CAF base</h1>

<p align="center">
<a href="https://pypi.org/project/caf.base/"><img alt="Supported Python versions" src="https://img.shields.io/pypi/pyversions/caf.base.svg?style=flat-square"></a>
<a href="https://pypi.org/project/caf.base/"><img alt="Latest release" src="https://img.shields.io/github/release/transport-for-the-north/caf.base.svg?style=flat-square&maxAge=86400"></a>
<a href="https://app.codecov.io/gh/Transport-for-the-North/caf.base"><img alt="Coverage" src="https://img.shields.io/codecov/c/github/transport-for-the-north/caf.base.svg?branch=master&style=flat-square&logo=CodeCov"></a>
<a href="https://github.com/Transport-for-the-North/caf.base/actions?query=event%3Apush"><img alt="Testing Badge" src="https://img.shields.io/github/actions/workflow/status/transport-for-the-north/caf.base/tests.yml?style=flat-square&logo=GitHub&label=Tests"></a>
</p>

<p align="center">
<a href="https://github.com/psf/black"><img alt="code style: black" src="https://img.shields.io/badge/code%20format-black-000000.svg"></a>
</p>

The Common Analytical Framework (CAF) Base (formerly caf.core) is a collection
of core classes and definitions used across the CAF framework of models and
tools. The goal of the CAF is to create easily implementable and adaptable
transport planning and appraisal functionalities. It's the beginning of a
project to make a lot of the useful stuff from
[NorMITs Demand](https://github.com/Transport-for-the-North/NorMITs-Demand)
more widely available and easily accessible.

Caf.base is a successor to the core module within NorMITs Demand. The class central
to caf.base is the DVector, which is used for storing and manipulating vectors
often used in transport modelling and analysis, such as trip end and land use data.
See documentation on the DVector, and it's constituent parts ZoningSystem and Segmentation
for more details.

## Contributing

CAF.base happily accepts contributions.
The best way to contribute to this project is to go to the issues tab and
report bugs or submit a feature request. This helps CAF.base become more
stable and full-featured. Please check the closed bugs before
submitting a bug report to see if your question has already been answered.
