"""
Power step implementations. They need `constraints` and `apply` functions implemented. 
"""
import tensorflow as tf

class ZeroPowerStep():
    
    def __init__(self):
        
        self.n_params = 0
        self.param_names = []
        
    def constraints(self,inputs,*args):
        
        return inputs
    
    def apply(self, t, *args):
        
        return 0


class Sawtooth():
    
    def __init__(self):
        
        self.n_params = 2
        self.param_names = ['q', 'r']
        
    def constraints(self,inputs, scalings=1):
        
        return inputs*scalings 
    
    def apply(self, t, q, r):

        
        return q + r*t
    

class FourSteps():
    
    def __init__(self, costum_dtype):
        
        self.n_params = 5
        self.dtype = costum_dtype
        self.param_names = ['q1','q2', 'q3', 'q4', 'r']
        
    def constraints(self,inputs, scalings=1):
        
        return inputs*scalings
    
    def apply(self, t, q1, q2, q3, q4, r):

        if tf.is_tensor(q1):
            step1 = q1*tf.cast(t>=0, self.dtype)
            step2 = q2*tf.cast(t>=900, self.dtype)
            step3 = q3*tf.cast(t>=1800, self.dtype)
            step4 = q4*tf.cast(t>=2700, self.dtype)
        else:
            step1 = q1*(t>0)
            step2 = q2*(t>=900)
            step3 = q3*(t>=1800)
            step4 = q4*(t>=2700)
        
        return step1 + step2 + step3 + step4 + r*t
    
   