# RSA-SCA: Specification Curve Analysis for Representational Similarity Analysis

## Overview
This repository contains the code and data for our study examining the reliability of representational similarity analysis (RSA) across different neuroimaging modalities through Specification Curve Analysis (SCA).

## Background
Computational neuromodeling methods present researchers with numerous analysis pipeline configurations. Researchers optimize neuroimaging processing pipelines to maximize end-to-end accuracy metrics and corresponding performance measures. However, the proliferation of degrees of freedom in analysis introduced by the complexity of these configuration decisions represents an underexamined source of bias in neuromodeling analyses.

Our work adapts Specification Curve Analysis (SCA), a principled method introduced to control for potential biases associated with degrees of freedom in psychological analyses, to neuroimaging data. We apply SCA to analysis specification choices that estimate representational dissimilarity matrices (RDMs) from fMRI and EEG activity to assess the reliability of inferences from representational similarity analysis.

## Repository Structure
This repository is organized as follows:
- `rsa_things/`:
  -  Source code for implementing the specification curve analysis in fMRI using [THINGS](https://elifesciences.org/articles/82580) dataset.
- `rsa_thingsEEG1/`:
  -  Source code for implementing the specification curve analysis in EEG using [EEG-based RSA analyses](https://www.nature.com/articles/s41597-021-01102-7).
- `rsa_gods/`:
  -  Code for implementing fMRI stat models using alternate pipelines on [GODS](https://www.nature.com/articles/ncomms15037) dataset.
  -  Also includes code for performing SCA on inter-model hypothesis tests from [this paper](https://www.sciencedirect.com/science/article/pii/S0893608022002982).



A preliminary version of this work was previously reported in [Sen Sarma, S., Boruah, G., & Srivastava, N. (2024)](https://escholarship.org/uc/item/62j3r1hq)

