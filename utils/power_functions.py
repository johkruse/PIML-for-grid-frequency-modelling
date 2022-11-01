"""
Power step implementations. They need `constraints` and `apply` functions implemented.  
"""


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
    

    
    
   