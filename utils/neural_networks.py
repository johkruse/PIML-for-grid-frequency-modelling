""" 
Constructors of artificial neural networks. 
"""

import tensorflow as tf


class MLP(tf.keras.Sequential):
    
    """Multilayer perceptron"""
    
    def __init__(self, layer_dims, activation_func, dropout_rate):
        super().__init__()
        
        for layer_dim in layer_dims[:-1]:
            self.add(tf.keras.layers.Dense(layer_dim,
                                           activation=activation_func))
            self.add(tf.keras.layers.Dropout(dropout_rate))
        
        self.add(tf.keras.layers.Dense(layer_dims[-1], activation='linear'))