"""
Implementation od data pre-processing routines for frequency data and techno-economic features
"""

import numpy as np
import pandas as pd
import tensorflow as tf


def load_data(freq_file = '../Frequency_data_preparation/TransnetBW/cleansed_2015-01-01_to_2019-12-31.h5',
              feature_folder = './data/CE/'):
    """
    Load target and feature data from file paths.

    Args:
        freq_file (str, optional): File path to frequency time series data.
        feature_folder (str, optional): Folder with feature data.

    Returns:
        tuple: frequency data and features
    """    
    
    # Read and normalize target
    freq = pd.read_hdf(freq_file)-50
    freq = freq *2*np.pi # transform to angular frequency
    
    # Read 15min resolved features
    features = pd.read_hdf(feature_folder+'input_actual.h5')
    features = features.join(pd.read_hdf(feature_folder+'input_forecast.h5'))

    # series has to start at 00:00
    assert freq.index[0].hour==0, 'frequency time series should start at 00:00'
    assert freq.index[0].minute==0, 'frequency time series should start at 00:00'
    
    return freq, features

def prepare_data(freq, features, prediction_start=0, n_prediction_steps=900, n_per_inteval = 900,
                 n_theta_0 = 60,  n_s_omega_0 = 60, custom_dtype = 'float64' ):
    """
    Prepare data frames of inputs, outputs and initial conditions. Outputs are created by reshaping the frequency
    time series into intervals, i.e., vectors, for each input instance. 

    Args:
        freq (pandas.Series): frequency data time series
        features (pandas.DataFrame): feature data
        prediction_start (int, optional): Start of predicted intervals in minutes before full hour.
        n_prediction_steps (int, optional): Actual number of predicted time steps within the interval.
        n_per_inteval (int, optional): Number of time steps per interval.
        n_theta_0 (int, optional): Number of time steps used to estimate theta initial condition.
        n_s_omega_0 (int, optional): Number of time steps used to estimate s_omega initial condition. 
        custom_dtype (str, optional): 'float64' or 'float32'.

    Returns:
        tuple: tensor with time steps, features, targets, initial conditions
    """            
    
    # Select start of predicted intervals 
    if prediction_start==0:
        freq_shifted = freq.iloc[3600-int(prediction_start*60):].copy()
    else:
        freq_shifted = freq.iloc[3600-int(prediction_start*60):-int(prediction_start*60)].copy()

    # Initialize time steps 
    dt = 1.  # time step
    interval_index = freq.index[3600::n_per_inteval]
    ts = tf.constant(np.arange(0,dt*n_prediction_steps, dt).astype(custom_dtype)) # time steps for ML model 

    # Prepare frequency data by reshaping
    Y = freq_shifted.values.astype(custom_dtype).reshape((freq_shifted.shape[0]//n_per_inteval,
                                                          n_per_inteval))
    Y = pd.DataFrame(data=Y[:,:n_prediction_steps],
                     index = interval_index,
                     columns=np.arange(n_prediction_steps))


    # Extract initial conditions     
    init_data = {
        'theta_0': freq_shifted.rolling(n_theta_0, closed='left').sum().iloc[::n_per_inteval].values*dt,
        'omega_0': Y.iloc[:,0].values, 
        's_theta_0': freq_shifted.cumsum().rolling(n_theta_0).std().iloc[::n_per_inteval].values*dt,
        'cov_theta_omega_0': np.zeros(Y.shape[0])+1e-10,
        's_omega_0': Y.iloc[:,:n_s_omega_0].std(1).values,
    }
    Z = pd.DataFrame(index = interval_index,
                     data=init_data,
                     dtype=custom_dtype)
    
    # Prepare features
    X = features.reindex(Y.index).astype(custom_dtype)
    X.loc[:,'hour_sin'] = np.sin((X.index.hour + X.index.minute/60)/24*2*np.pi)
    X.loc[:,'hour_cos'] = np.cos((X.index.hour + X.index.minute/60)/24*2*np.pi)
    X.loc[:,'minute_sin'] = np.sin((X.index.minute)/60*2*np.pi)
    X.loc[:,'minute_cos'] = np.cos((X.index.minute)/60*2*np.pi)
    X = X.drop(columns=['month', 'hour', 'weekday'])

    # Remove intervals with NaNs
    valid_ind = ~pd.concat([X, Y, Z], axis=1).isnull().any(axis=1)
    Y,X, Z = Y[valid_ind], X[valid_ind], Z[valid_ind]


    assert ~(Z.s_omega_0==0).any(), "s_omega should not contain zeros"
    assert ~(Z.s_theta_0==0).any(), "s_theta should not contain zeros"

    return ts, X,Y,Z
