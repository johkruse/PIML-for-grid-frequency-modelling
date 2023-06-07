import sys

sys.path.append('./')

import pickle
import warnings
from datetime import datetime

import keras_tuner as kt
import numpy as np
import pandas as pd
import shap
import tensorflow as tf
from utils import data_prep
from utils.ml_models import SOHyperModel
from utils.power_functions import FourSteps


# Setup the version tag
version= '2023-05-03'

startTime = datetime.now()
custom_dtype='float64'
tf.keras.backend.set_floatx(custom_dtype)
 
# Define loss
def negloglik(p, rv_p):
    return -rv_p.log_prob(p)
  
# Prepare target/feature data and initial conditions Z
freq, features = data_prep.load_data()
ts, dfs = data_prep.prepare_data(freq, features, custom_dtype=custom_dtype) 

# Define features for ex-post model and day-ahead model
day_ahead_cols = dfs['X_train'].filter(regex='day_ahead|hour|minute').columns # day-ahead forecasts and hour
expost_cols = dfs['X_train'].drop(columns=day_ahead_cols).columns.append(pd.Index(['prices_day_ahead'])) # only ex-post and prices


#HPO for ex-post/day-ahead model with different prediction length
for feature_types, cols in zip(['day_ahead','ex_post'], [ day_ahead_cols,expost_cols]):

    print('\n############ Training {} model ##########\n'.format(feature_types))
    
    model_dir = "./results/CE/version_{}/".format(version)+feature_types+'/'
    
    # Early stopping callback definition
    stop_early = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10,
                                                  min_delta=1, restore_best_weights=1,
                                                  mode='min')
    

    # Fixed hyper-parameters
    fixed_model_hps = data_prep.prepare_fixed_hps(features=dfs['X_train'].loc[:,cols], ts=ts,
                                                  power_step= FourSteps(custom_dtype),
                                                  param_scalings=[1,0.01,0.1,100,100,0.01,0.001,0.001,0.001,0.001, 0.000001],
                                                  vmins=[0.001, 0, 0.001, 10, 30, 0.0001, 0,0,0,0,0])


    # Define a Second-order model with hyper-parameters
    hyper_model = SOHyperModel(**fixed_model_hps, loss=negloglik)   
    
    # Initialize a hyper-parameter tuner
    tuner = kt.RandomSearch(
        hyper_model,
        objective=kt.Objective('val_loss', direction='min'),
        max_trials=10,
        executions_per_trial=1,
        overwrite=True,
        directory=model_dir,
        project_name="tuning"
    )

    #Tune hyper-parameters and use early stopping in ANN training
    print('\n#### Hyper-parameter search ####\n')
    tuner.search([dfs['X_train'].loc[:,cols], dfs['Z_train']], dfs['y_train'],
                 epochs=100, batch_size=128,
                 validation_data=([dfs['X_val'].loc[:,cols], dfs['Z_val']], dfs['y_val']),
                 callbacks=[stop_early])
    
    # Find best epoch for best HPs
    print('\n#### Fitting and saving best model ####\n')
    best_hps=tuner.get_best_hyperparameters(num_trials=1)[0]
    model = tuner.hypermodel.build(best_hps)
    history = model.fit([dfs['X_train'].loc[:,cols], dfs['Z_train']], dfs['y_train'],
                        epochs=200, batch_size=128,
                        validation_data=([dfs['X_val'].loc[:,cols], dfs['Z_val']], dfs['y_val']))
    val_loss_per_epoch = history.history['val_loss']
    best_epoch = val_loss_per_epoch.index(min(val_loss_per_epoch)) + 1
    print('Best epoch:', best_epoch)

    # Fit best model on train and validation set
    model = tuner.hypermodel.build(best_hps)
    history = model.fit([dfs['X_train'].append(dfs['X_val']).loc[:,cols], dfs['Z_train'].append(dfs['Z_val'])],
                        dfs['y_train'].append(dfs['y_val']),
                        epochs=best_epoch, batch_size=128)
    
    # Save best model 
    model.save_weights(model_dir + 'best_model/')
    with open(model_dir + 'fixed_model_hps.pkl', 'wb') as f:
        pickle.dump(fixed_model_hps, f)
        
        
    # Additional analysis for ex-post model 
    if feature_types=='ex_post':
                    
        
        # SHAP values for best ex-post model
        print('\n#### SHAP values ####\n')
        X_test_to_explain = dfs['X_test'].loc[:,cols]
        model.apply_scaling = False # Turn off scaling to ensure numerical stability of SHAP
        shap_exp = shap.KernelExplainer(model.call_params,
                                        dfs['X_train'].loc[:,cols].sample(200).values)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            raw_shap_explanations = shap_exp.shap_values(X_test_to_explain)
        for i,param in enumerate(model.st_process.param_names[2:]):
            shap_vals = pd.DataFrame(index=X_test_to_explain.index,
                                     columns=X_test_to_explain.columns,
                                     data=raw_shap_explanations[i])
            shap_vals = shap_vals*model.constraint_and_scaling.scalings.numpy()[i] # Re-introduce correct scaling
            shap_vals.to_hdf(model_dir+'shap_values_{}_long.h5'.format(param), key='df')
        model.apply_scaling = True # Turn scaling back on
                
        # Dependency on scaling variations
        print('\n#### Scaling dependency calculation ####\n')
        for init_seed in range(10):
            tf.random.set_seed(init_seed)
            for kappa_scale in [1000,100]:
                for r_scale in [1e-6,1e-5]:
                    for q_scale in [1e-3,1e-2]:
                        for D_scale in [0.01,0.1]: 
                            pscalings = tf.constant([1,np.nan,0.1,
                                                    np.nan,kappa_scale,
                                                    D_scale,q_scale,q_scale,q_scale,q_scale, r_scale],
                                                    dtype=custom_dtype)
                            tuner.hypermodel.param_scalings=pscalings
                            model = tuner.hypermodel.build(best_hps)
                            history = model.fit([dfs['X_train'].loc[:,cols], dfs['Z_train']],
                                                dfs['y_train'],
                                                epochs=100, batch_size=128,
                                                validation_data=([dfs['X_val'].loc[:,cols], dfs['Z_val']], 
                                                                 dfs['y_val']),
                                                callbacks=[stop_early])
                            model_scaling_dir = model_dir + 'best_model_kappa{}_q{}_r{}_D{}_seed{}/'.format(kappa_scale,
                                                                                                            q_scale,
                                                                                                            r_scale,
                                                                                                            D_scale,
                                                                                                            init_seed)
                            model.save_weights(model_scaling_dir)
                            np.savetxt(model_scaling_dir + 'n_epochs.txt',  [len(history.history['loss'])])
                            
            # Effect of no scaling            
            pscalings = tf.constant([1,np.nan,1,np.nan,1,1,1,1,1,1, 1],
                        dtype=custom_dtype)
            tuner.hypermodel.param_scalings=pscalings
            model = tuner.hypermodel.build(best_hps)
            history = model.fit([dfs['X_train'].loc[:,cols], dfs['Z_train']], dfs['y_train'],
                                epochs=100, batch_size=128,
                                validation_data=([dfs['X_val'].loc[:,cols], dfs['Z_val']],
                                                 dfs['y_val']),
                                callbacks=[stop_early])
            model.save_weights(model_dir + 'best_model_no_scaling_seed{}/'.format(init_seed))
            np.savetxt(model_dir + 'best_model_no_scaling_seed{}/n_epochs.txt'.format(init_seed),
                        [len(history.history['loss'])])



print('\nTotal execution time:', datetime.now() - startTime)



