""" 
High-level model pipelines grid frequency dynamics
"""

import keras_tuner as kt
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

from utils.constraints import SOContstraintScaling
from utils.neural_networks import MLP
from utils.stochastic_processes import SOPowerStep, SOSawtooth


class DailyProfileModel(tf.keras.Model):
    
    """Daily profile predictor with time-dependent Gaussian noise"""

    def __init__(self):
        super(DailyProfileModel, self).__init__()
        self.dp_means = None
        self.dp_stds = None
        
        self.prob1 =  tfp.layers.DistributionLambda(
            lambda t: tfp.distributions.MultivariateNormalDiag(loc= t[...,0],
                                                               scale_diag= t[...,1])
            )
    
    def fit(self, y_train):
        """ 
        Fit daily profile model. 

        Args:
            y_train (pandas.DataFrame): Training data of shape (n_batch, n_time_steps). 

        """     
        
        self.dp_means = y_train.groupby(y_train.index.time).mean()
        self.dp_stds = y_train.groupby(y_train.index.time).std()
        
    
    def compile(self):
        pass
    
    def call(self, data):
        """ 
        Call daily profile predictor.

        Args:
            data (pandas.DataFrame): test data. The prediction is constructed for the DateTimeIndex of this data.

        Returns:
            tfp.distributions: tensorflow probabilistic distribution
        """
        index_test=data.index
        
        pred_mean =  np.array([self.dp_means.loc[time] for time in index_test.time])
        pred_std = np.array([self.dp_stds.loc[time] for time in index_test.time])
    
        dp_input = np.concatenate([pred_mean[...,np.newaxis],
                                   pred_std[...,np.newaxis]],
                                  axis=-1)


        return self.prob1(dp_input)
        

class ConstantModel(tf.keras.Model):
    
    """Constant predictor with time-independent Gaussian noise """

    def __init__(self):
        super(ConstantModel, self).__init__()
        self.c_means = None
        self.c_stds = None
        
        self.prob1 =  tfp.layers.DistributionLambda(
            lambda t: tfp.distributions.MultivariateNormalDiag(loc= t[...,0],
                                                               scale_diag= t[...,1])
            )
    
    def fit(self, y_train):
        """ 
        Fit a constant model with time-independent means and std. devs. 

        Args:
            y_train (pandas.DataFrame): Training data of shape (n_intervals, n_time_steps). 

        """     
        
        self.c_means = np.repeat(y_train.values.flatten().mean(), y_train.shape[1])
        self.c_stds = np.repeat(y_train.values.flatten().std(), y_train.shape[1])
        
    
    def compile(self):
        pass
    
    def call(self, data):
        """ 
        Call constant predictor.

        Args:
            data (pandas.DataFrame): test data. The prediction is constructed for the DateTimeIndex of this data.

        Returns:
            tfp.distributions: tensorflow probabilistic distribution
        """
        
        index_test=data.index
        
        pred_mean =  np.array([self.c_means for ind in index_test])
        pred_std = np.array([self.c_stds for ind in index_test])
    
        dp_input = np.concatenate([pred_mean[...,np.newaxis],
                                   pred_std[...,np.newaxis]],
                                  axis=-1)


        return self.prob1(dp_input)

    
class SOModel(tf.keras.Model):
    
    """Model pipeline for second-order stochastic models with arbitrary power steps"""
    
    def __init__(self, ts, power_step,vmin, param_scalings, feature_mean=0,
                 feature_var=1, list_of_units = [128,128, 64, 16],
                 activation_func='tanh', dropout_rate=0, use_sawtooth_analyt=True,
                 apply_scaling=True):
        """
        Initialize the model.

        Args:
            ts (tensor):  1-d tensor with time steps of one interval.
            power_step (class): Power step implementation. Needs `constraints` and `apply` method.
            vmin (tensor): 1-d tensor of shape (n_params) with minimum values for parameters, which enter the constraint layer. 
            param_scalings (tensor): 1-d tensor of shape (n_params) with parameter scalings. 
            Outputs of the ML model are multiplied with these scalings. Defaults to 1.
            feature_mean (int, optional): Temporal mean of features for normalization. Should be from training set.
            Defaults to 0.
            feature_var (int, optional): Temporal variance of features for normalization. Should be from training set.
            Defaults to 1.
            list_of_units (list, optional): List with number of units per layer of the neural network.
            Defaults to [128,128, 64, 16].
            activation_func (str, optional): Activation function. Defaults to 'tanh'.
            dropout_rate (int, optional): Dropout rate. Defaults to 0.
            use_sawtooth_analyt (bool, optional): If True, use the `SOSawtooth` class if sawtooth power steps are used. Otherwise, use
            `SOPowerSteps` in all cases. Defaults to True.
            apply_scaling (bool, optional): If False, revert the parameter scaling in the parameter model. If True, scaling is applied
            without intervention. 
        """

        
        super().__init__()
        
        # Determine number of predicted initial conditions and total number of predicted parameters
        self.n_init_cond = 3
        self.n_params = 3 + self.n_init_cond + power_step.n_params
        self.apply_scaling=apply_scaling
        
        # Define layers of model pipeline
        self.input_normalization = tf.keras.layers.Normalization(mean=feature_mean,
                                                                 variance=feature_var)
        self.nn = MLP( list_of_units +  [self.n_params ], activation_func, dropout_rate)
        self.constraint_and_scaling = SOContstraintScaling(power_step, vmin, param_scalings)
        if use_sawtooth_analyt and (power_step.__class__.__name__=='Sawtooth'):
            self.st_process = SOSawtooth(ts)
        else:
            self.st_process = SOPowerStep(ts, power_step)
        self.prob_dist = tfp.layers.DistributionLambda(
            lambda t: tfp.distributions.MultivariateNormalDiag(loc= t[...,0], scale_diag= t[...,1])
            )
        
    def generate_sample(self, inputs, rolling=True):
        """ 
        Generate randomly sampled trajectories from parameters. 
        
        Args: 
            inputs (tensor): Inputs of model pipeline. 
            rolling (bool): If true, the last value of a sample interval is used as initial condition for the next interval.
            ...
        Returns:
            nd.array: sample trajectories 
        """   
        x = inputs[0]
        y0 = tf.gather(inputs[1], indices=[0,1], axis=-1)
        
        x = self.call_params(x)      
        x = tf.keras.layers.Concatenate()([y0, x])
        
        return self.st_process.generate_sample(x, rolling)
        
    def call_params(self, features):
        """ 
        Call the parameters model.

        Args:
            features (tensor): Features with operational data of shape (n_batch, n_featurs), without data of initial
            conditions. 

        Returns:
            tensor: Predicted model parameters with shape (n_batch, n_params).
        """
        
        x = self.input_normalization(features)
        x = self.nn(x)
        x = self.constraint_and_scaling(x)

        if not self.apply_scaling:
            return x/self.constraint_and_scaling.scalings
        else:
            return x
    
    def call(self, inputs):
        """ 
        Call the whole probabilistic model.

        Args:
            inputs (list): List [X, Z] with a tensor X containing features and a tensor Z containing initial conditions. 
            X should have shape (n_batch, n_features) and Z should have shape (n_batch, 5), i.e., Z includes all initial
            conditions ['theta_0','omega_0','s_theta_0','cov_theta_omega_0', 's_omega_0'], which were estimated
            from the data. However, we only use the initial means from Z in the model.

        Returns:
            tfp.distributions: tensorflow probabilistic distribution
        """

        x = inputs[0]
        y0 = tf.gather(inputs[1], indices=[0,1], axis=-1)
        
        x = self.call_params(x)       
        x = tf.keras.layers.Concatenate()([y0, x])
        x = self.st_process(x) 
        x = self.prob_dist(x)
        
        return x
    

class SOHyperModel(kt.HyperModel):
    
    """Meta-model for hyper-parameter tunig using the second-order stochastic model with arbitrary power steps"""
    
    def __init__(self, vmin,  param_scalings, power_step, ts, feature_mean,
                 feature_var, loss):
        super().__init__()
        self.dtype=ts.dtype
        self.param_scalings = param_scalings
        self.vmin = vmin
        self.power_func=power_step
        self.ts = ts
        self.feature_means=feature_mean
        self.feature_vars=feature_var
        self.loss=loss
    
    def build(self, hp):
    
        learning_rate = hp.Choice("lr", [1e-4, 1e-3, 1e-2])
        dropout_rate = hp.Choice("dropout", [0.,0.1,0.2,0.3])
        n_units = hp.Choice("n_units", [64,128])
        list_of_units = [n_units for i in range(hp.Choice("n_layers", [2,4,6]))] 
        activation_func = hp.Choice('activation_func', ['tanh', 'sigmoid'])

        model = SOModel(self.ts, self.power_func, self.vmin, self.param_scalings,
                        feature_mean=self.feature_means, feature_var=self.feature_vars,
                        list_of_units=list_of_units, activation_func=activation_func,
                        dropout_rate=dropout_rate)

        model.compile(optimizer=tf.optimizers.Adam(learning_rate=learning_rate),
                      loss=self.loss)
        
        return model

