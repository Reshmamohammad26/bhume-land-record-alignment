# BhuMe Geospatial AI Take-Home Assignment

## Overview

This repository contains my solution for the **BhuMe Geospatial AI Take-Home Assignment**, which focuses on aligning cadastral land parcel boundaries with real-world field boundaries visible in satellite imagery.

The objective is to identify and correct boundary misalignments caused by historical surveying and georeferencing inaccuracies while providing calibrated confidence scores and exercising restraint when corrections cannot be made reliably.

---

## Problem Statement

Official cadastral boundaries often exhibit spatial drift when overlaid on modern satellite imagery. The challenge is to:

* Detect boundary misalignment
* Correct plot geometries where reliable evidence exists
* Assign meaningful confidence scores
* Flag uncertain cases instead of forcing corrections

The final output is a `predictions.geojson` file containing corrected or flagged plot boundaries for each village.

---

## Dataset

Each village dataset contains:

### input.geojson

Official cadastral plot boundaries and associated metadata.

### imagery.tif

High-resolution satellite imagery used as the primary visual reference.

### boundaries.tif

Machine-generated field boundary hints that assist alignment.

### example_truths.geojson

A small set of manually validated plot corrections used for evaluation and benchmarking.

---

## Approach

### 1. Baseline Analysis

The provided starter kit baseline uses a global median shift derived from example truth plots. While effective for correcting large-scale drift, it does not account for local distortions or village-specific variations.

### 2. Local Alignment Strategy

This solution extends the baseline by:

* Extracting local image patches around plots
* Analyzing imagery and boundary hints
* Estimating plot-level alignment adjustments
* Applying village-specific alignment logic

Separate alignment modules are implemented for:

* Vadnerbhairav
* Malatavadi

to accommodate differences in parcel density and landscape characteristics.

### 3. Prediction Generation

For each plot:

* Estimate alignment quality
* Generate corrected geometry when appropriate
* Preserve original geometry when confidence is insufficient
* Export predictions in the required GeoJSON format

---

## Confidence Calibration Strategy

Confidence values are designed to reflect the reliability of each correction rather than serving as arbitrary scores.

Confidence is influenced by:

* Alignment quality
* Boundary agreement
* Local imagery consistency
* Geometric plausibility

The goal is to ensure that higher confidence predictions correspond to more accurate corrections, improving calibration and ranking performance.

---

## Repository Structure

```text
your-repo/
├── quickstart.py
├── bhume/
│   ├── __init__.py
│   ├── io.py
│   ├── geo.py  
│   ├── baseline.py
│   ├── aligner.py        ← Vadnerbhairav aligner
│   ├── pipeline.py
│   └── score.py
├── malatavadi/
│   └── aligner_mal.py    ← Malatavadi-tuned aligner
├── predictions/
│   ├── vadnerbhairav/
│   │   └── predictions.geojson
│   └── malatavadi/
│       └── predictions.geojson
└── transcripts/
    └── README.md         
```

---


## Usage

Run the starter workflow:

```bash
python quickstart.py data/vadnerbhairav
```

Generate predictions:

```bash
python -m bhume.pipeline
```

Evaluate predictions:

```bash
python -m bhume.score
```

---

## Results Summary

This project focuses on improving upon the provided baseline through localized alignment strategies and confidence-aware decision making.

Key objectives:

* Improve boundary alignment accuracy
* Improve confidence calibration
* Maintain restraint by avoiding unsupported corrections
* Generalize across multiple villages without manual editing

---

## AI Usage Statement

AI tools were used to assist with:

* Understanding the problem domain
* Reviewing geospatial alignment strategies
* Designing confidence calibration approaches
* Improving repository organization
* Code review and documentation

All AI-assisted conversations used during development are documented in the `transcripts/` directory as requested by the assignment guidelines.

---

## Future Improvements

Potential enhancements include:

* Plot-level adaptive alignment
* Advanced image-based contour extraction
* Learned confidence calibration models
* Affine and non-rigid local transformations
* Automated validation using neighboring parcel consistency

---

## Author

**Mohammad Reshma**


