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


def prepare_data(freq, features, add_time_features=True,
                 prediction_start=0, n_prediction_steps=3600, n_per_inteval = 3600,
                 n_theta_0 = 60,  n_s_omega_0 = 60, custom_dtype = 'float64',
                 train_end='2017-12-31 23:59', val_end='2018-12-31 23:59',
                 test_end='2019-12-31 23:59'):
    """
    Prepare data frames of inputs, outputs and initial conditions. Outputs are created by reshaping the frequency
    time series into intervals, i.e., vectors, for each input instance. 

    Args:
        freq (pandas.Series): frequency data time series.
        features (pandas.DataFrame): feature data.
        add_time_features (bool, optional): Whether to add hour and minute features.
        prediction_start (int, optional): Start of predicted intervals in minutes before full hour.
        n_prediction_steps (int, optional): Actual number of predicted time steps within the interval.
        n_per_inteval (int, optional): Number of time steps per interval.
        n_theta_0 (int, optional): Number of time steps used to estimate theta initial condition.
        n_s_omega_0 (int, optional): Number of time steps used to estimate s_omega initial condition. 
        custom_dtype (str, optional): 'float64' or 'float32'.
        train_end (string, optional): End of training set (format 'Year-Month-Day HH:MM')
        val_end (string, optional): End of validation set.
        test_end (string, optional): End of test set.
        

    Returns:
        tuple: tensor with time steps, dictionary of prepared data
    """            
    
    # Select start of predicted intervals 
    assert prediction_start<15, 'Prediction start has to be smaller than 15 min!'
    if prediction_start==0:
        freq_shifted = freq.iloc[3600:].copy()
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
    X = X.drop(columns=['month', 'hour', 'weekday'], errors='ignore')
    if add_time_features:
        X.loc[:,'hour_sin'] = np.sin((X.index.hour + X.index.minute/60)/24*2*np.pi)
        X.loc[:,'hour_cos'] = np.cos((X.index.hour + X.index.minute/60)/24*2*np.pi)
        if n_per_inteval<3600:
            X.loc[:,'minute_sin'] = np.sin((X.index.minute)/60*2*np.pi)
            X.loc[:,'minute_cos'] = np.cos((X.index.minute)/60*2*np.pi)

    # Remove intervals with NaNs
    valid_ind = ~pd.concat([X, Y, Z], axis=1).isnull().any(axis=1)
    Y,X, Z = Y[valid_ind], X[valid_ind], Z[valid_ind]


    assert ~(Z.s_omega_0==0).any(), "s_omega should not contain zeros"
    assert ~(Z.s_theta_0==0).any(), "s_theta should not contain zeros"

    # Split train, test and validation set
    X_train = X.loc[:train_end].sample(frac=1)
    Z_train = Z.loc[X_train.index]
    y_train = Y.loc[X_train.index]
    
    X_val = X.loc[train_end:val_end].sample(frac=1)
    Z_val = Z.loc[X_val.index]
    y_val = Y.loc[X_val.index]
    
    X_test = X.loc[val_end:test_end].sample(frac=1)
    Z_test = Z.loc[X_test.index]
    y_test = Y.loc[X_test.index]    
    
    data = {'X_train':X_train, 'Z_train':Z_train, 'y_train':y_train,
            'X_val':X_val, 'Z_val':Z_val, 'y_val':y_val,
            'X_test':X_test, 'Z_test':Z_test, 'y_test':y_test}
    
    return ts, data



def prepare_fixed_hps(features,
                      ts,
                      param_scalings,
                      power_step,
                      vmins,
                      custom_dtype='float64'):
    """ 
    Prepare fixed hyperparameters (not optimized) for PIML model.

    Args:
        features (pandas.DataFrame): Feature data . Should be from training set.
        ts (tensor): 1-d tensor with time steps of one interval.
        param_scalings (list): List of scaling parameters for dynamical system parameters and
        initial conditions. Note that scaling can also be defined for cov_0 and tau, but they are not not applied!
        power_step (function): Deterministic power imbalance function. 
        vmins (list): List of minimum values for dynamical system parameters and 
        initial conditions. The minimum values have to fullfill the physical constraints of the parameters.
        custom_dtype (str, optional): Custom float precision. 

    Returns:
        dict: Dictionary of train, validation and test data
    """


    fixed_model_hps = {
        'param_scalings': tf.constant(param_scalings, dtype=custom_dtype), 
        'power_step':power_step,
        'vmin':tf.constant(vmins, dtype=custom_dtype),  
        'ts':ts,
        'feature_mean':features.mean(0).values,#needs to by numpy array!
        'feature_var':features.var(0).values,#needs to by numpy array!
    }

    return fixed_model_hps