# Estimation and Approximation Problems Over Groups (0372-4013) - Final Project 

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](????) [![MIT License](https://img.shields.io/apm/l/atomic-design-ui.svg?)](https://github.com/tterb/atomic-design-ui/blob/master/LICENSEs)![Run Unit-Tests](https://github.com/RedCrow9564/EstimationOverGroups-FinalProject/workflows/Run%20Unit-Tests/badge.svg)![Compute Code Metrics](https://github.com/RedCrow9564/EstimationOverGroups-FinalProject/workflows/Compute%20Code%20Metrics/badge.svg)![GitHub last commit](https://img.shields.io/github/last-commit/RedCrow9564/EstimationOverGroups-FinalProject)

This is a project submitted as a requirement for this course. [The course](https://www30.tau.ac.il/yedion/syllabus.asp?course=0372401301) was administered in Spring 2020 in [Tel-Aviv University - School of Mathematical Sciences](https://en-exact-sciences.tau.ac.il/math), and taught by [Dr. Nir Sharon](https://en-exact-sciences.tau.ac.il/profile/nsharon). 
This project is a reconstruction of experiments of [[1]](#1) for an algorithm for the Multi-Reference Factor Analysis problem. A complete documentation of the code is available [here](????)(open it in a web browser).

## Getting Started

The code can be fetched from [this repo](https://github.com/RedCrow9564/EstimationOverGroups-FinalProject.git). The Jupyter Notebook version does the same work, and can be deployed to [Google Colab](?????). While the the notebook version can be used immediately, this code has some prerequisites.
Any questions about this project may be sent by mail to 'eladeatah' at mail.tau.ac.il (replace 'at' by @).

### Prerequisites

This code was developed for Windows10 OS and tested using the following Python 3.7.6 dependencies. These dependencies are listed in [requirements.txt](requirements.txt).
All these packages can be installed using the 'pip' package manager (when the command window is in the main directory where requirements.txt is located):
```
pip install -r requirements.txt
```
All the packages, except for Sacred, are available as well using 'conda' package manager.

## Running the Unit-Tests

The Unit-Test files are:

* [test_diagonal_extraction_and_construction.py](UnitTests/test_diagonal_extraction_and_construction.py) - Tests for components of the first stage of the algorithm: diagonals estimation and estimator construction.
* [test_phase_retrieval_actions.py](UnitTests/test_phase_retrieval_actions.py) - Tests for components of the second stage of the algorithm: coefficients matrix construction and Fourier basis transition.
* [test_tri_spectrum_estimation.py](UnitTests/test_tri_spectrum_estimation.py) - Tests for the tri-spectrum's estimation methods.

Running any of these tests can be performed by:
```
<python_path> -m unittest <test_file_path>
```
## Acknowledgments
Credits for the original algorithms, paper and results of [[1]](#1) belong to its respectful authors: [Prof. Yoel Shkolnisky](https://en-exact-sciences.tau.ac.il/profile/yoelsh) and Dr. Boris Landa.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details

## References
<a id="1">[1]</a> [B. Landa, Y. Shkolnisky. Multi-reference factor analysis: low-rank covariance estimation
under unknown translations (arXiv: 2019, expected 2020-2021)](https://arxiv.org/pdf/1906.00211.pdf).

