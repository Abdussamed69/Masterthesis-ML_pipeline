# ML_pipeline
Olive oil classification

# Olive Oil Classification Using Spectroscopy and Machine Learning

This repository contains the Python software developed as part of my Master's thesis in **Electrical Systems Engineering** at Heilbronn University in collaboration with **INSION GmbH**.

The project investigates the classification of olive oil samples using **fluorescence spectroscopy** and **VIS/NIR spectroscopy** combined with machine learning.

The complete processing chain covers:

**Spectral measurement → Data preprocessing → Feature matrix generation → Machine learning training → Model validation → Classification of new measurements**

---

## Project Objective

The objective of this work is to classify olive oil samples based on their optical spectral characteristics.

The measurement system records:

* Fluorescence spectra using UV excitation at **275 nm**
* Fluorescence spectra using UV excitation at **325 nm**
* Fluorescence spectra using UV excitation at **365 nm**
* VIS reflection spectra using a broadband light source

The measured spectra are processed in Python and used as input for supervised machine learning models.

The developed workflow enables the classification of different olive oil classes as well as adulterated oil samples.

---

## System Overview

The experimental setup consists of:

* INSION VIS microspectrometer
* INSION NIR microspectrometer
* SDCM4 spectrometer controller
* LED driver
* UV LEDs
* Broadband light source
* Cuvette holder and optical measurement head
* PC-based Python measurement software

The SDCM4 controller communicates with the computer and controls the spectral measurements. The acquired detector data are transferred to the computer, decoded and stored for further processing.

---

## Software Architecture

The project is mainly based on five Python scripts.

| Script                 | Function                                                                                                               |
| ---------------------- | ---------------------------------------------------------------------------------------------------------------------- |
| `SDCM.py`              | Implements classes and functions for communication with the INSION SDCM4 spectrometer controller                       |
| `Fluorem_v3.py`        | Measurement software with GUI for controlling measurements and storing spectral data                                   |
| `preprocessing.py`     | Processes individual measurement files, calculates reflection values, extracts metadata and generates feature matrices |
| `ML_pipeline.py`       | Performs preprocessing, normalization, model training, validation, hyperparameter optimization and model storage       |
| `ClassificationApp.py` | Loads trained models and classifies newly acquired olive oil measurements                                              |

---

## Processing Workflow

### 1. Data Acquisition

Spectral measurements are performed using `Fluorem_v3.py`.

The software communicates with the SDCM4 controller through functions implemented in `SDCM.py`.

Each individual measurement is stored as an Excel file.

The file names contain metadata such as:

* Timestamp
* Measurement system identifier
* Sample origin
* Olive oil class

Example:

```text
20251018-114437_P.224-990056_Arbequina_extra virgin olive oil.xlsx
```

---

### 2. Preprocessing

The script `preprocessing.py` processes the individual measurement files.

Main preprocessing steps include:

* Reading spectral measurement files
* Extracting class labels from filenames
* Extracting sample origin information
* Removing irrelevant spectral regions
* Calculation of normalized reflection spectra
* Combining measurements into a feature matrix
* Visualization of spectra

The resulting feature matrix follows the standard supervised learning structure:

```text
Measurement | Feature 1 | Feature 2 | ... | Feature n | Label
```

Each row represents one spectral measurement.

---

### 3. Machine Learning Pipeline

The script `ML_pipeline.py` prepares the feature matrix for machine learning.

The main processing steps are:

1. Separation of features `X` and labels `y`
2. Removal of irrelevant data
3. Train/test splitting
4. Z-score normalization using `StandardScaler`
5. Training of classification models
6. Cross-validation
7. Hyperparameter optimization using `GridSearchCV`
8. Calculation of evaluation metrics
9. Storage of trained models and scalers

Several classification algorithms were evaluated during the project.

The final models are stored using `joblib` and can subsequently be used for new measurements.

---

## Model Validation

Model performance is evaluated using established classification metrics, including:

* Accuracy
* Precision
* Recall
* F1-score
* Confusion matrix
* Cross-validation score

Cross-validation and independent test data are used to evaluate the generalization capability of the trained models.

---

## Classification of New Samples

`ClassificationApp.py` applies the trained models to new spectral measurements.

For each measurement channel, the application:

1. Loads the corresponding trained model
2. Loads the associated scaler
3. Imports the new spectral measurement
4. Applies the same normalization used during training
5. Calculates class probabilities
6. Determines the predicted olive oil class
7. Outputs the classification result and confidence value

Example output:

```text
Predicted class: extra virgin olive oil
Confidence: 0.98
```

The prediction results can also be exported as Excel files.

---

## Main Python Libraries

The project uses several Python libraries for measurement control, data processing and machine learning, including:

```text
numpy
pandas
scikit-learn
joblib
matplotlib
openpyxl
```

Additional libraries may be required depending on the measurement-system configuration and GUI implementation.

---

## Repository Structure

A simplified project structure may look like:

```text
Master-Thesis/
│
├── SDCM.py
├── Fluorem_v3.py
├── preprocessing.py
├── ML_pipeline.py
├── ClassificationApp.py
│
├── data/
│   ├── 275nm/
│   ├── 325nm/
│   ├── 365nm/
│   └── lamp_VIS/
│
├── ML/
│   ├── models/
│   └── new_measurements/
│
├── figures/
│
└── README.md
```

Measurement data, trained models and directory paths may need to be adapted to the local system before execution.

---

## Scientific Background

The project combines three main disciplines:

* Optical spectroscopy
* Data processing
* Machine learning

Instead of directly determining all standardized chemical quality parameters of olive oil, the optical measurement system evaluates characteristic spectral patterns and surrogate parameters.

Machine learning is subsequently used to identify relationships between these spectral characteristics and the predefined sample classes.

---

## Master Thesis

**Title:**
*Classification of Olive Oil Samples Based on Spectral Surrogate Parameters Using Fluorescence and VIS/NIR Spectroscopy and Machine Learning*

**Degree:**
Master of Science – Electrical Systems Engineering

**Institution:**
Heilbronn University

**Industrial Partner:**
INSION GmbH

---

## Notes

This repository represents a research and prototype implementation developed during a Master's thesis.

The software is intended for scientific and engineering purposes and does not replace standardized laboratory methods defined by the European Union or the International Olive Council for official olive oil quality assessment.

Hardware-specific functions may require an INSION spectrometer system and SDCM4 controller.

---

## Author

**Abdussamed Korkmaz**
M.Sc. Electrical Systems Engineering
Heilbronn University
