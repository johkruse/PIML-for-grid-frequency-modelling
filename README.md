# Physics-informed machine learning for power grid frequency modelling

Code accompanying the paper "Physics-informed machine learning for power grid frequency modelling".
Preprint: https://doi.org/10.48550/arXiv.2211.01481

## Install

The code is written in Python (tested with python 3.7). To install the required dependencies execute the following commands:

```bash
python3.7 -m venv ./venv
source ./venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

## Usage

The `scripts` folder contains the script to produce the paper results and `notebooks` contains a notebook to reproduce the paper figures. Moreover, it includes a notebook with a code tutorial, which gives an overview over the different code modules and their usage. The `utils` folder comprises modules with custom classes and functions such as the physics-informed machine learning (PIML) model.

## Input data and results

All the input data is publicly available and we have uploaded our results on zenodo. The data and the (intermediate) results can be used to run the scripts.

* **Techno-economic features**: The feature data `input_forecast.h5` and `input_actual.h5` were taken from [this data repository](https://doi.org/10.5281/zenodo.5118351). The data is assumed to reside in the repository directory within `./data/CE/`
* **Results of hyper-parameter optimization and model interpretation**: The output of the script `./scripts/model_fit.py` is available on [this repository](https://doi.org/10.5281/zenodo.8014065) and it is assumed to reside in `./results/CE/`.  
* **Raw grid frequency data**: We have used pre-processed [grid frequency data](https://doi.org/10.5281/zenodo.5105820) to train our model. The repository provides yearly CSV files. To use them in this code, the yearly data has to be concatenated and saved as a HDF file with the path "../Frequency_data_preparation/TransnetBW/cleansed_2015-01-01_to_2019-12-31.h5" relative to this code repository. The frequency data is originally based on publicly available measurements from [TransnetBW](https://www.transnetbw.de/de/strommarkt/systemdienstleistungen/regelenergie-bedarf-und-abruf).
