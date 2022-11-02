# Physics-inspired machine learning for power grid frequency modelling

Code accompanying the manuscript "Physics-inspired machine learning for power grid frequency modelling".
Preprint: (inserted later)

## Install

The code is written in Python (tested with python 3.7). To install the required dependencies execute the following commands:

```bash
python3.7 -m venv ./venv
source ./venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

## Usage

The `scripts` folder contains scripts to create the paper results and `notebooks` contains a notebook to reproduce the paper figures. The `utils` folder comprises modules with custom classes and functions such as the physics-inspired machine learning (PIML) model.

The `scripts` contain a pipeline of four different stages:

* `1_download_data.sh`: A bash script to download the external features from the ENTSO-E Transparency Platform.
* `2_entsoe_data_prep.py`: Collect and aggregate external features.
* `3_external_feature_prep.py`: Add additional engineered features to the set of external features.
* `4_model_fit.py`: Fit the PIML model, optimize hyper-parameters and calculate SHAP values.

## Input data and results

All the raw data is publicly available and we have uploaded the processed data and our results on zenodo. The data and the (intermediate) results can be used to run the scripts.

* **External features and results of hyper-parameter optimization and model interpretation**: The output of scripts 2 to 4 are available on [zenodo](https://zenodo.org/record/xxxx). The data is assumed to reside in the repository directory within `./data/` and the results should reside in `./results/`. 
* **Raw grid frequency data**: We have used pre-processed [grid frequency data](https://zenodo.org/record/5105820) to train our model. The repository provides yearly CSV files. To use them in this code, the yearly data has to be concatenated and saved as a HDF file with the path "../Frequency_data_preparation/TransnetBW/cleansed_2015-01-01_to_2019-12-31.h5" relative to this code repository. The frequency data is originally based on publicly available measurements from [TransnetBW](https://www.transnetbw.de/de/strommarkt/systemdienstleistungen/regelenergie-bedarf-und-abruf).
* **Raw ENTSO-E data**: The output of `1_download_data.sh` is not available on the zenodo repository, but can be downloaded from the [ENTSO-E Transparency Platform](transparency.entsoe.eu/) via the bash script. The ENTSO-E data is assumed to reside in `../../External_data/ENTSO-E` relative to this code repository.
