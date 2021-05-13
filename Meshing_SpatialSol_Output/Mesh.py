# Import packages
import csv # Read in case parameters and write out solutions
import numpy as np
from scipy import sparse

class case_param(): 
    def __init__(self,param): 
        self.name = param['case_name'] # now the name is given inside the case, not as the case's actual name 
        self.dim = 1 # dimensions 
        self.x0 = 0.0 # inlet position 
        self.xL = self.x0 + float(param['length']) # outlet 
        self.dx = float(param['dx'])
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

class mesh():
    def __init__(self,case): # Take in the case info for certain params
        dim = 1 # case.dim
        if (dim == 1):
            self.Nx = int((case.xL - case.x0)/case.dx + 2.0)
        
            # Face locations
            self.xc = np.ones(self.Nx)*case.x0# Initialize mesh
            self.xc[self.Nx-1] = case.xL # Outward boundary
            for i in range(2,self.Nx-1): 
                self.xc[i] = (i-1)*case.dx # Cell Face Locations

            # Node locations
            self.x = self.xc # Initialize mesh
            for i in range(0,self.Nx-1):
                self.x[i] = (self.xc[i+1] + self.xc[i])/2 # Cell Node Locations: halfway between faces
            self.x[self.Nx-1] = self.xc[self.Nx-1] # Outward boundary
    
    def output(self,fname): # output mesh
        with open(fname,'w', newline='') as csvfile:
            mesh_write = csv.writer(csvfile, delimiter = '\t')
            mesh_write.writerow(['i', 'x', 'xc'])
            for i in range(0,self.Nx):
                mesh_write.writerow([i,self.x[i],self.xc[i]])

with open('casefile.csv',newline='') as casefile: 
    casereader = csv.DictReader(casefile) 
    i = 0 
    caselist = {} 
    for row in casereader:    
        caselist[i] = row 
        i += 1         
caselist[0]['dx']
base = case_param(caselist[0]) # base case object
print(base.x0)
print(base.dx)

base_mesh = mesh(base) # base mesh object

Nx = base_mesh.Nx # too much text for commonly used variable

# Cursory Check
print(base_mesh.x[0:5])
print(Nx)
print(base_mesh.x[Nx])

base_mesh.output('base_mesh.dat') # Output mesh to file for full confirmation





