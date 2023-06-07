""" 
Implementation of the stochastic differential equations and their solution. 
"""

import tensorflow as tf
import numpy as np
from tqdm import tqdm

    
class SOPowerStep(tf.keras.layers.Layer):
        
    def __init__(self, ts, power_step, dt_ratio=10):
        """
        Second-order stochastic process with arbitrary deterministic power function. 

        Args:
            ts (tensor): 1-d tensor with time steps of one interval.
            power_step (class): Power step implementation. Needs `constraints` and `apply` method.   
            dt_ratio (int, optional): Ratio of time deltas dt_numeric/dt_output between dt used for numerical integration
            and dt used for display/output (as defined in ts). In this model dt_ratio is used
            for sample generation and integration of power steps. Defaults to 1.
        """
        
        super().__init__(trainable=False)
        self.ts = tf.expand_dims(ts, axis=-1) # needs shape (n_time_steps, 1)
        self.param_names = ['theta_0','omega_0','s_theta_0','cov_theta_omega_0', 's_omega_0', # 0-4 is y0
                            'tau', 'kappa', 'D'] + power_step.param_names 
        self.dt_ratio = dt_ratio # ratio between output time steps and number of time steps for numerical integration
        self.dt = (ts[...,1]-ts[...,0]) / dt_ratio # dt for numerical integration
        self.ts_integ = tf.expand_dims(np.arange(ts[0],ts[-1]+self.dt, self.dt),  # time array for numerical integration
                                       axis=-1) 
        self.power_step = power_step
        self.custom_dtype = ts.dtype
    
       
    def _check_tensor(self, *args):
        """
        Internal function to check whether input tensors have correct shape. Inputs and outputs should have
        shape (n_batch,...). To calculate the mean/std.devation for single float values (e.g. for inspection),
        this function automatically adds a batch dimension to ensure compatibility. 
        
        Args:
            *args: Float, tensor or list of floats/tensors

        Returns:
            list: Compatible tensors
        """
        
        checked_args = []
        
        for arg in args:
            if tf.is_tensor(arg):
                if len(arg.shape)==1:
                    checked_args += [tf.expand_dims(tf.cast(arg, dtype=self.custom_dtype), axis=0)]
                else:
                    checked_args += [tf.cast(arg, dtype=self.custom_dtype)]
            else:
                checked_args += [tf.expand_dims(tf.constant(arg, dtype=self.custom_dtype), axis=0)]
        
        return checked_args  
    
    def generate_sample(self, params_in, rolling=True):
        """ 
        Generate randomly sampled trajectories from parameters. 
        
        Args: 
            params (tf.tensor or nd.array): entries are in same order as `self.param_names`. 
            Should have shape (n_intervals, n_params), where n_intervals is the number of 
            intervals for which a trajectory with fixed parameters is generated.
            rolling (bool): If true, the last value of a sample interval is used as initial condition for the next interval.
            The initial conditions from `params` (for the other intervals) are ignored.
            ...
        Returns:
            nd.array: sample trajectories 
        """    
        if type(params_in)!=np.ndarray:
            params = np.array(params_in)
        else:
            params = params_in.copy()
        n_intervals = params.shape[0]
        n_steps = self.ts_integ.shape[0]
        sqrtdt = np.sqrt(self.dt)
        dt = self.dt.numpy()
        
        Y = np.zeros((n_intervals,2, n_steps))
        
        if not rolling:        
        
            # Fix initial conditions of each interval 
            Y[:,:,0] = params[:,:2]
            
            # Numerical integration (intervals are processed in parallel)
            for i in range(n_steps-1):
                
                Y[:,0,i + 1] = Y[:,0,i] + dt * Y[:,1,i] 
                Y[:,1,i + 1] = (Y[:,1,i] + dt * (-Y[:,1,i] / params[:,5] - Y[:,0,i]/params[:,6]**2 
                                                    + self.power_step.apply(i*dt, *list(params[:,8:].T)) 
                                                    ) 
                                        + params[:,7] * sqrtdt * np.random.randn(n_intervals)
                                )
        else:
            
            # Fix initial condition only for first interval
            Y[0,:,0] = params[0,:2]
            
            # Rolling numerical integration for each interval
            for j in tqdm(range(n_intervals)):
                for i in range(n_steps-1):
                
                    Y[j,0,i + 1] = Y[j,0,i] + dt * Y[j,1,i] 
                    Y[j,1,i + 1] = (Y[j,1,i] + dt * (-Y[j,1,i] / params[j,5] - Y[j,0,i]/params[j,6]**2 
                                                        + self.power_step.apply(i*dt, *list(params[j,8:].T)) 
                                                        ) 
                                            + params[j,7] * sqrtdt * np.random.randn()
                                    )
                # Choose last values as initial condition for next interval
                if j<n_intervals-1:
                    Y[j+1,1,0] = Y[j,1,-1]
                    Y[j+1,0,0] = Y[j,1,-60:].sum(-1)*dt # to reproduce how we estimate theta_0 during model fitting
        
        return Y[:, 1, ::self.dt_ratio]
      
    def mean_t(self, y0, tau, kappa, power_params, check_tensor=True):
        """
        Calculate mean trajectory for interval with semi-analytical solution.

        Args:
            y0 (tensor): initial condition with columns y0 = [theta_0, omega_0]
            tau (tensor): tau
            kappa (tensor): kappa
            power_params (list): list of tensors that represent the power step parameters
            check_tensor (bool, optional): When True, check and correct the shapes of input tensors to exhibit 
            (n_batch,...) shape. Defaults to True.

        Returns:
            tensor: trajectory with shape (n_natch, n_time_steps)
        """   
        
        if check_tensor:
            y0, tau, kappa = self._check_tensor(y0, tau, kappa)
            power_params = self._check_tensor(*power_params)
        
        # eigenvalues
        l0 = (-tf.sqrt(1- 4*tau**2/kappa**2)-1) / (2*tau)
        l1 = (tf.sqrt(1-4*tau**2/kappa**2)-1) / (2*tau)
        
        # integrate on a time scale that is reduced by dt_ratio
        integrand0 = self.power_step.apply(self.ts_integ, *power_params)*tf.exp(-l0*self.ts_integ)* self.dt 
        integrand1 = self.power_step.apply(self.ts_integ, *power_params)*tf.exp(-l1*self.ts_integ)* self.dt 
        
        omega_mean = (
            #homogeneous part
            (y0[...,0]/(kappa**2)*(tf.exp(l1*self.ts)- tf.exp(l0*self.ts)) +
             y0[...,1]*(l0*tf.exp(l0*self.ts)-l1*tf.exp(l1*self.ts))) / (l0-l1) + 
            # inhomogeneous part
            (l0*tf.exp(l0*self.ts)*tf.math.cumsum(integrand0,exclusive=True, axis=-2)[::self.dt_ratio] -
             l1*tf.exp(l1*self.ts)*tf.math.cumsum(integrand1,exclusive=True, axis=-2)[::self.dt_ratio])  / (l0-l1)
        )  
        

        return tf.transpose(omega_mean)
    
    
    def sigma_t(self, y0, tau, kappa, D, check_tensor=True):
        """
        Calculate trajectory of std.dev. for interval.

        Args:
            y0 (tensor): initial condition with columns y0 = (s_theta, cov_theta_omega, s_omega)
            tau (tensor): tau
            kappa (tensor): kappa
            D (tensor): D
            check_tensor (bool, optional): When True, check and correct the shapes of input tensors to exhibit 
            (n_batch,...) shape. Defaults to True.

        Returns:
            tensor: trajectory with shape (n_natch, n_time_steps)
        """
        
        if check_tensor:
            y0, tau, kappa, D = self._check_tensor(y0, tau, kappa, D)
        
        # eigenvalues
        l0 = -1/tau
        l1 = (- tf.sqrt(1-4*tau**2/kappa**2)-1)/tau
        l2 = (tf.sqrt(1-4*tau**2/kappa**2)-1)/tau

        s_omega_2 = (
            # homogeneous part
            (y0[..., 0]**2*(tau**2/kappa**2)*(tf.exp(l1*self.ts) +
                                              tf.exp(l2*self.ts) -
                                              tf.exp(l0*self.ts)*2) +
             y0[..., 1]*(tf.exp(l0*self.ts)*(-2*tau) +
                         tf.exp(l1*self.ts)*((8*tau**2/kappa**2)*(l1*kappa**2/(4*tau)+1)/(l1-l2)) +
                         tf.exp(l2*self.ts)*((8*tau**2/kappa**2)*(l2*kappa**2/(4*tau)+1)/(l2-l1))) +
             y0[..., 2]**2*(tf.exp(l0*self.ts)*(-2*tau**2) +
                            tf.exp(l1*self.ts)*((2*l1*tau**2-l1*kappa**2-2*tau)/(l2-l1)) +
                            tf.exp(l2*self.ts)*((2*l2*tau**2-l2*kappa**2-2*tau)/(l1-l2)))) / (kappa**2-4*tau**2) +
            # inhomogeneous part
            (2*tau**2/l0 * (1-tf.exp(l0*self.ts)) +
             (2*tau/l1-2*tau**2+kappa**2) / (l2-l1) * (1-tf.exp(l1*self.ts)) +
             (2*tau/l2-2*tau**2+kappa**2) / (l1-l2) * (1-tf.exp(l2*self.ts))) * D**2 / (kappa**2 - 4*tau**2)
        )

        return tf.transpose( tf.sqrt(s_omega_2) )
    
    
    def call(self, params):
        """ 
        Calculate means and std. deviations for this process. 
        
        Args: 
            params (tensor): entries are in same order as `self.param_names`. 
            ...
        Returns:
            tensor: stacked trajectories of means and std. deviations with shape (n_batch, n_time_steps, 2)
        """     
        
        mean_vals = self.mean_t(params[...,:2],
                                params[...,5],
                                params[...,6],
                                tf.unstack(params[...,8:], axis=-1),
                                check_tensor=False
                                )
        
        std_vals = self.sigma_t(params[...,2:5], 
                                params[...,5],
                                params[...,6],
                                params[..., 7],
                                check_tensor=False
                                )

        return tf.stack([mean_vals, std_vals], axis=-1)
    
class SOSawtooth(tf.keras.layers.Layer):
    
    
    def __init__(self, ts, dt_ratio=1):
        """
        Second-order stochastic process with sawtooth power steps.

        Args:
            ts (tensor): 1-d tensor with time steps of one interval.
            dt_ratio (int, optional): Ratio of time deltas dt_numeric/dt_output between dt used for numerical integration
            and dt used for display/output (as defined in ts). In this model dt_ratio is only used
            for sample generation. Defaults to 1.
        """
        
        super().__init__(trainable=False)
        self.ts = tf.expand_dims(ts, axis=-1) # needs shape (n_time_steps, 1) for tensor broadcasting
        self.param_names = ['theta_0','omega_0','s_theta_0','cov_theta_omega_0', 's_omega_0', # 0-4 is y0
                            'tau', 'kappa', 'D', 'q', 'r'] 
        self.dt_ratio = dt_ratio # ratio between number of time steps for numerical integration and output time steps
        self.dt = (ts[...,1]-ts[...,0]) / dt_ratio # dt for numerical integration
        self.ts_integ = tf.expand_dims(np.arange(ts[0],ts[-1]+self.dt, self.dt),  # time array for numerical integration
                                       axis=-1) 
        self.custom_dtype = ts.dtype
       
    def _check_tensor(self, *args):
        """
        Internal function to check whether input tensors have correct shape. Inputs and outputs should have
        shape (n_batch,...). To calculate the mean/std.deviation for single float values (e.g. for inspection),
        this function automatically adds a batch dimension to ensure compatibility. 
        
        Args:
            *args: Float, tensor or list of floats/tensors

        Returns:
            list: Compatible tensors
        """
        
        
        checked_args = []
        
        for arg in args:
            if tf.is_tensor(arg):
                if len(arg.shape)==1:
                    checked_args += [tf.expand_dims(tf.cast(arg, dtype=self.custom_dtype), axis=0)]
                else:
                    checked_args += [tf.cast(arg, dtype=self.custom_dtype)]
            else:
                checked_args += [tf.expand_dims(tf.constant(arg, dtype=self.custom_dtype), axis=0)]
        
        return checked_args  
        
    def generate_sample(self, params_in, rolling=True):
        """ 
        Generate randomly sampled trajectories from parameters. 
        
        Args: 
            params_in (tf.tensor or nd.array): entries are in same order as `self.param_names`. 
            Should have shape (n_intervals, n_params), where n_intervals is the number of 
            intervals for which a trajectory with fixed parameters is generated.
            rolling (bool): If true, the last value of a sample interval is used as initial condition for the next interval.
            The initial conditions from `params_in` (for the other intervals) are ignored.
            ...
        Returns:
            nd.array: sample trajectories 
        """    
        
        # Initialize values and make sure that all arrays are numpy arrays (to avoid costly type conversion)
        if type(params_in)!=np.ndarray:
            params = np.array(params_in)
        else:
            params = params_in.copy()
        n_intervals = params.shape[0]
        n_steps = self.ts_integ.shape[0]
        sqrtdt = np.sqrt(self.dt)
        dt = self.dt.numpy()
        
        Y = np.zeros((n_intervals,2, n_steps))
        
        if not rolling:        
        
            # Fix initial conditions of each interval 
            Y[:,:,0] = params[:,:2]
            
            # Numerical integration (intervals are processed in parallel)
            for i in range(n_steps-1):
                
                Y[:,0,i + 1] = Y[:,0,i] + dt * Y[:,1,i] 
                Y[:,1,i + 1] = (Y[:,1,i] + dt * (-Y[:,1,i] / params[:,5] - Y[:,0,i]/params[:,6]**2 
                                                    + (params[:,8]+params[:,9]*i*dt) 
                                                    ) 
                                        + params[:,7] * sqrtdt * np.random.randn(n_intervals)
                                )
        else:
            
            # Fix initial condition only for first interval
            Y[0,:,0] = params[0,:2]
            
            # Rolling numerical integration for each interval
            for j in tqdm(range(n_intervals)):
                for i in range(n_steps-1):
                
                    Y[j,0,i + 1] = Y[j,0,i] + dt * Y[j,1,i] 
                    Y[j,1,i + 1] = (Y[j,1,i] + dt * (-Y[j,1,i] / params[j,5] - Y[j,0,i]/params[j,6]**2 
                                                        + (params[j,8]+params[j,9]*i*dt) 
                                                        ) 
                                            + params[j,7] * sqrtdt * np.random.randn()
                                    )
                # Choose last values as initial condition for next interval
                if j<n_intervals-1:
                    Y[j+1,1,0] = Y[j,1,-1]
                    Y[j+1,0,0] = Y[j,1,-60:].sum(-1)*dt # to reproduce of we estimate theta_0 during model fitting
                    
        return Y[:, 1, ::self.dt_ratio]
    
    
    def mean_t(self, y0, tau, kappa, q,r , check_tensor=True):
        """
        Calculate mean trajectory for interval.

        Args:
            y0 (tensor): initial condition with columns y0 = [theta_0, omega_0]
            tau (tensor): tau
            kappa (tensor): kappa
            q (tensor): q
            r (tensor): r
            check_tensor (bool, optional): When True, check and correct the shapes of input tensors to exhibit 
            (n_batch,...) shape. Defaults to True.

        Returns:
            tensor: trajectory with shape (n_natch, n_time_steps)
        """      
        
        if check_tensor:
            y0, tau, kappa = self._check_tensor(y0, tau, kappa)
        
        # eigenvalues
        l0 = (-tf.sqrt(1- 4*tau**2/kappa**2)-1) / (2*tau)
        l1 = (tf.sqrt(1-4*tau**2/kappa**2)-1) / (2*tau)

        omega_mean = (
            #homogeneous part
            (y0[...,0]/kappa**2*(tf.exp(l1*self.ts)- tf.exp(l0*self.ts)) +
             y0[...,1]*(l0*tf.exp(l0*self.ts)-l1*tf.exp(l1*self.ts))) / (l0-l1) + 
            # inhomogeneous part
            (tf.exp(l0*self.ts)*(q+r/l0) - tf.exp(l1*self.ts)*(q+r/l1) +
             r*(1/l1-1/l0))  / (l0-l1)
        )  
        
        return tf.transpose(omega_mean)
    
    def sigma_t(self, y0, tau, kappa, D, check_tensor=True):
        """
        Calculate trajectory of std.dev. for interval.

        Args:
            y0 (tensor): initial condition with columns y0 = (s_theta, cov_theta_omega, s_omega)
            tau (tensor): tau
            kappa (tensor): kappa
            D (tensor): D
            check_tensor (bool, optional): When True, check and correct the shapes of input tensors to exhibit 
            (n_batch,...) shape. Defaults to True.

        Returns:
            tensor: trajectory with shape (n_natch, n_time_steps)
        """
        
        if check_tensor:
            y0, tau, kappa, D = self._check_tensor(y0, tau, kappa, D)
        
        # eigenvalues
        l0 = -1/tau
        l1 = (- tf.sqrt(1-4*tau**2/kappa**2)-1)/tau
        l2 = (tf.sqrt(1-4*tau**2/kappa**2)-1)/tau

        s_omega_2 = (
            # homogeneous part
            (y0[..., 0]**2*(tau**2/kappa**2)*(tf.exp(l1*self.ts) +
                                              tf.exp(l2*self.ts) -
                                              tf.exp(l0*self.ts)*2) +
             y0[..., 1]*(tf.exp(l0*self.ts)*(-2*tau) +
                         tf.exp(l1*self.ts)*((8*tau**2/kappa**2)*(l1*kappa**2/(4*tau)+1)/(l1-l2)) +
                         tf.exp(l2*self.ts)*((8*tau**2/kappa**2)*(l2*kappa**2/(4*tau)+1)/(l2-l1))) +
             y0[..., 2]**2*(tf.exp(l0*self.ts)*(-2*tau**2) +
                            tf.exp(l1*self.ts)*((2*l1*tau**2-l1*kappa**2-2*tau)/(l2-l1)) +
                            tf.exp(l2*self.ts)*((2*l2*tau**2-l2*kappa**2-2*tau)/(l1-l2)))) / (kappa**2-4*tau**2) +
            # inhomogeneous part
            (2*tau**2/l0 * (1-tf.exp(l0*self.ts)) +
             (2*tau/l1-2*tau**2+kappa**2) / (l2-l1) * (1-tf.exp(l1*self.ts)) +
             (2*tau/l2-2*tau**2+kappa**2) / (l1-l2) * (1-tf.exp(l2*self.ts))) * D**2 / (kappa**2 - 4*tau**2)
        )

        return tf.transpose( tf.sqrt(s_omega_2) ) 
    
    
    def call(self, params):
        """ 
        Calculate means and std. deviations for this process. 
        
        Args: 
            params (tensor): entries are in same order as `self.param_names`. 
            ...
        Returns:
            tensor: stacked trajectories of means and std. deviations with shape (n_batch, n_time_steps, 2)
        """     
        
        mean_vals = self.mean_t(params[...,:2],
                                params[...,5],
                                params[...,6],
                                params[...,8],
                                params[...,9],
                                check_tensor=False
                                )
        

        std_vals = self.sigma_t(params[...,2:5], 
                                params[...,5],
                                params[...,6],
                                params[..., 7],
                                check_tensor=False
                                )

        return tf.stack([mean_vals, std_vals], axis=-1)
    
