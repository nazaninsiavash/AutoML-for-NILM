
# AutoML4NILM

## Installation

To get started with AutoML4NILM, follow these steps:

### 1. Install NILMTK

First, you need to install NILMTK. **Important:** Do not follow the manual installation instructions from the main NILMTK repository, as they can lead to dependency conflicts.

The easiest and most reliable way to install NILMTK is:

```bash
pip install nilmtk_s14pe
```

### 2. Install Additional Dependencies

After installing NILMTK, install the additional required packages listed in the `requirements.txt` file:

```bash
pip install -r requirements.txt
```

## About the Project

This project has been tested with the UK-DALE dataset but is designed to be easily extendable to other datasets as well.

The AutoML framework currently supports 11 machine learning algorithms for NILM (Non-Intrusive Load Monitoring). Thanks to its flexible and modular design, the framework can be further enhanced to support more algorithms, models, or datasets in the future.


## Paper

This repository supports the following paper:

> **N. Siavash and A. Moin, “A Bayesian Optimization-Based AutoML Framework for Non-Intrusive Load Monitoring,”**  
> *arXiv preprint arXiv:2602.05739*, 2026.

### Citation

```bibtex
@article{SiavashMoin2026,
  title   = {A Bayesian Optimization-Based AutoML Framework for Non-Intrusive Load Monitoring},
  author  = {Siavash, Nazanin and Moin, Armin},
  journal = {arXiv preprint arXiv:2602.05739},
  year    = {2026}
}
````

---

## Keywords

Non-intrusive load monitoring, NILM, automated machine learning, AutoML, Bayesian optimization, energy disaggregation, machine learning, smart meters, energy consumption, and time-series analysis.

## Acknowledgements

This project builds upon the following repositories:

- [autonialm](https://github.com/ukritw/autonialm)
- [NILMTK](https://github.com/nilmtk/nilmtk)
- [NILMTK-Contrib](https://github.com/nilmtk/nilmtk-contrib)

We gratefully acknowledge their contributions and codebases.




