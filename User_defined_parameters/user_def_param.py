import csv

class case_param(): 
    def __init__(self,param): 
        self.name = param['case_name'] # now the name is given inside the case, not as the case's actual name 
        self.dim = 1 # dimensions 
        self.x0 = 0.0 # inlet position 
        self.xL = self.x0 + float(param['length']) # outlet 
        fluid_name = param['fluid'] 
        mu = float(param['mu']) 
        u0 = 0.0 
        p0 = float(param['p0']) # inlet pressure 
        pL = float(param['pL']) # outlet 
        self.fl = {'Name': fluid_name, 'mu': mu, 'u0': u0, 'p0': p0, 'pL': pL} 
        pm_name = param['porous_medium']  
        K = float(param['K']) 
        eps = float(param['eps']) 
        self.pm = {'Name': pm_name, 'K':K, 'eps':eps} 
        self.fl['u0'] = -K/mu*(pL-p0)/(self.xL-self.x0)

with open('casefile.csv',newline='') as casefile: 
    casereader = csv.DictReader(casefile) 
    i = 0 
    caselist = {} 
    for row in casereader:    
        caselist[i] = row 
        print(row['case_name'], row['fluid'], row['mu']) # check that code works as expected 
        i += 1         
        
        
base = case_param(caselist[0]) 
oil = case_param(caselist[4]) 
print('Base flow velocity (m/s): ' + str(base.fl['u0'])) 
print('Oil flow velocity (m/s): ' + str(oil.fl['u0'])) 