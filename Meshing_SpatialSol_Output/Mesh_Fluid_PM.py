# Import packages
import csv # Read in case parameters and write out solutions
import numpy as np

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
            self.x = np.copy(self.xc) # Initialize mesh
            for i in range(0,self.Nx-1):
                self.x[i] = (self.xc[i+1] + self.xc[i])/2 # Cell Node Locations: halfway between faces
            self.x[self.Nx-1] = np.copy(self.xc[self.Nx-1]) # Outward boundary
    
    def output(self,fname): # output mesh
        with open(fname,'w', newline='') as csvfile:
            mesh_write = csv.writer(csvfile,dialect = 'excel', delimiter = '\t') # writer object
            mesh_write.writerow(['i', 'x', 'xc']) # header row
            for i in range(0,self.Nx):
                mesh_write.writerow([i+1,self.x[i],self.xc[i]]) # actual data rows
                
class fluid():
    def __init__(self,mesh,fluid_prop):
        self.name = fluid_prop['Name']
        # Initialize variables
        self.p = np.ones(mesh.Nx)*fluid_prop['p0'] # Pressure
        self.p[mesh.Nx-1] = fluid_prop['pL'] # Pressure boundary at x = L
        self.u = np.ones(mesh.Nx)*fluid_prop['u0'] # Velocity: Staggered mesh so velocity at faces
        self.mu = np.ones(mesh.Nx)*fluid_prop['mu'] # Viscosity
    def p_lin(self,mesh):
        N = mesh.Nx
        L = mesh.x[N-1]
        L0 = mesh.x[0]
        for i in range(1,N):
            self.p[i] = (self.p[N-1]-self.p[0])/(L-L0)*mesh.x[i]
    def darcyv(self,mesh,pm):
        N = mesh.Nx
        self.u[0] = -pm.K[0]/self.mu[0]*(self.p[1]-self.p[0])/(mesh.x[1]-mesh.x[0]) # inlet
        self.u[1] = self.u[0] # same location
        for i in range(2,N-1): # interior faces
            Ai = pm.K[i-1]/self.mu[i-1]/(mesh.xc[i]-mesh.x[i-1])
            Ai1 = pm.K[i]/self.mu[i]/(mesh.x[i]-mesh.xc[i])
            self.u[i] = -Ai*Ai1/(Ai+Ai1)*(self.p[i]-self.p[i-1])
        self.u[N-1] = -pm.K[N-1]/self.mu[N-1]*(self.p[N-1]-self.p[N-2])/(mesh.x[N-1]-mesh.x[N-2]) # outlet

class por_med():
    def __init__(self,mesh,pm_prop):
        self.name = pm_prop
        # Initialize Variables
        self.K = np.ones(mesh.Nx)*pm_prop['K'] # Permeability
        self.eps = np.ones(mesh.Nx)*pm_prop['eps'] # Porosity
#-----------------------------------------------------------------

with open('casefile.csv',newline='') as casefile:  # load cases
    casereader = csv.DictReader(casefile) 
    i = 0 
    caselist = {} 
    for row in casereader:    
        caselist[i] = row 
        i += 1         
caselist[0]['dx']
base = case_param(caselist[0])
print('Inlet:',base.x0)
print('Spacing:',base.dx)

base_mesh = mesh(base) # base mesh object

Nx = base_mesh.Nx # too much text for commonly used variable

# Cursory Check
print('Inlet and node spacing:',base_mesh.x[0:5])
print('Number of elements:',Nx)
print('Outlet:',base_mesh.x[Nx-1])

base_mesh.output('base_mesh.dat') # Output mesh to file for full confirmation

fl1 = fluid(base_mesh,base.fl)
pm1 = por_med(base_mesh,base.pm)

print('Original pressure (0):',fl1.p[0:4])
fl1.p_lin(base_mesh)
print('Linear pressure:',fl1.p[0:4])

print('Original Velocity (correct):',fl1.u[0:4])
fl1.u = np.zeros(base_mesh.Nx)
fl1.darcyv(base_mesh,pm1)
print('Darcy Velocity:',fl1.u[0:4])