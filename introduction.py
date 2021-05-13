class case_param(): 
    def __init__(self): 
        self.dim = 1 # dimensions 
        self.x0 = 0.0 # inlet position 
        self.xL = 1.0 # outlet 
        fluid_name = 'Water' 
        mu = 0.001 
        u0 = 0.0 
        p0 = 0.0 # inlet pressure 
        pL = -100.0 # outlet 
        self.fl = {'Name': fluid_name, 'mu': mu, 'u0': u0, 'p0': p0, 'pL': pL} 
        pm_name = 'Sand' 
        K = 1.0E-9 
        eps = 0.15 
        self.pm = {'Name': pm_name, 'K':K, 'eps':eps} 
        self.fl['u0'] = -K/mu*(pL-p0)/(self.xL-self.x0) 

base = case_param() 
base.u0 = -base.pm['K']/base.fl['mu']*(base.fl['pL']-base.fl['p0'])/(base.xL-base.x0) 

print(base.u0) 
print(base.fl['u0']) 