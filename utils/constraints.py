""" 
Implementation of parameter constraints for stochastic processes. 
"""

import tensorflow as tf
  
    
class SOContstraintScaling(tf.keras.layers.Layer):
    
    """
    Implementation of parameter constraints and scalings for second-order stochastic processes with arbitrary power step
    """
    
    def __init__(self, power_step, vmin, param_scalings):
        """
        Initialize constraints and scalings.

        Args:
            power_step (class): Power step implementation. 
            vmin (tensor): 1-d tensor with minimum values of each parameter.  
            param_scalings (tensor). 1-d tensor with parameter scalings.
        """
        super().__init__(trainable=False)
        
        self.power_step = power_step
        self.vmin = vmin
        self.scalings = param_scalings
    
    def call(self, inputs):
        """
        Transform inputs such that outputs obey the parameter constraints and exhibit given scale.

        Args:
            inputs (tensor): Has shape (n_batch, n_columns) and columns represent ['s_theta_0','cov_theta_omega_0',
            's_omega_0', 'tau', 'kappa', 'D'] + columns of power step parameters.

        Returns:
            tensor: Transformed inputs.
        """        

        s_theta_0 = tf.math.softplus(inputs[...,0])*self.scalings[0] + self.vmin[0]
        s_omega_0 = tf.math.softplus(inputs[...,2])*self.scalings[2] + self.vmin[2]
        cov_theta_omega_0 = 0.999*tf.tanh(inputs[...,1])*s_theta_0*s_omega_0
        kappa = tf.math.softplus(inputs[...,4])*self.scalings[4] + self.vmin[4]
        tau = (2*self.vmin[3]/kappa + 0.999*tf.sigmoid(inputs[...,3])*(1-2*self.vmin[3]/kappa))*kappa/2
        D = tf.math.softplus(inputs[...,5])*self.scalings[5] + self.vmin[5]
        params = tf.stack([s_theta_0, cov_theta_omega_0, s_omega_0, tau, kappa, D], axis=-1)


        # Apply constraints and scalings to parameters of power step
        power_params = self.power_step.constraints(inputs[..., params.shape[-1]:],
                                                   scalings=self.scalings[params.shape[-1]:])
        
        return tf.concat([params, power_params], axis=-1)
    