# Import packages
import csv # Read in case parameters and write out solutions
import numpy as np # all the array stuff
import matplotlib.pyplot as plt # plotting

class case_param(): # case parameter class
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
        pL = float(param['pL']) # outlet pressure
        self.fl = {'Name': fluid_name, 'mu': mu, 'u0': u0, 'p0': p0, 'pL': pL} 
        pm_name = param['porous_medium']  
        K = float(param['K']) 
        eps = float(param['eps']) 
        self.pm = {'Name': pm_name, 'K':K, 'eps':eps} 
        self.fl['u0'] = -K/mu*(pL-p0)/(self.xL-self.x0)

class mesh(): # mesh class
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

class fluid(): # fluid class, can create multiple fluid objects for multiphase flow or other studies
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

class por_med(): # porous medium class, for parametric studies or composite porous media
    def __init__(self,mesh,pm_prop):
        self.name = pm_prop
        # Initialize Variables
        self.K = np.ones(mesh.Nx)*pm_prop['K'] # Permeability
        self.eps = np.ones(mesh.Nx)*pm_prop['eps'] # Porosity

def plot_out(data): # plotting function, takes in data object of specific form and prints plots
    N_param = data.Np
    N = data.Nx
    fig, ax = plt.subplots(N_param,1,figsize=(4,5))    
    for i in range(0,N_param):
        ax[i].plot(data.x, data.var[0:,i], color = 'black',linewidth=0.5)
        ax[i].set_xlabel(data.varnamex,fontsize=12) 
        ax[i].set_ylabel(data.varname[i],fontsize=12)
    fig.tight_layout()
    plt.show()    
# ----------------------------------------------------------------------#

## Reading the case parameters from a csv file
with open('casefile.csv',newline='') as casefile: # opening the casefile csv
    casereader = csv.DictReader(casefile) 
    i = 0 
    caselist = {} 
    for row in casereader:    
        caselist[i] = row 
        i += 1         
caselist[0]['dx']
base = case_param(caselist[0])
print(base.x0)
print(base.dx)

## Initialize and check mesh object creation ##
base_mesh = mesh(base) # create base mesh from case parameters
print('Node Locations w/ inlet:', base_mesh.x[0:5]) # check inlet location and spacing
print('Nx:', base_mesh.Nx) # check number of elements
print('Outlet Location:', base_mesh.x[base_mesh.Nx-1])
print('Face Locations:', base_mesh.xc[0:5]) #

## Create fluid and porous medium objects for this specific case ##
fl1 = fluid(base_mesh,base.fl) # fluid object, determined by mesh and case's fluid properties
pm1 = por_med(base_mesh,base.pm) # porous medium object, determined by mesh and case's porous medium properties

## Calculate pressure using simple linear function ##
print('Initial Pressure:',fl1.p[0:4])
fl1.p_lin(base_mesh)
print('Linear Pressure:',fl1.p[0:4])

## Calculate velocity ##
print('Initial Velocity (correct):',fl1.u[0:4]) # velocity from initialization
fl1.u = np.zeros(base_mesh.Nx) # zero out velocity 
fl1.darcyv(base_mesh,pm1) # use darcyv method
print('Final Velocity:',fl1.u[0:4]) # print to confirm that darcyv did what it was supposed to (got same solution as initialization)


# Data/solution object to be plotted #
data_sol = type('sol', (object,), {})() # empty object (metaprogramming)

# Data at nodes: x p, K, mu
data_sol.Np = 3 # p, K, mu
data_sol.Nx = base_mesh.Nx
data_sol.varnamex = 'x (m)'
data_sol.varname = ['p (Pa)', 'K ($m^2$)', '\u03BC (Pa*s)']
data_sol.x = base_mesh.x
data_sol.var = np.zeros((data_sol.Nx,data_sol.Np))
data_sol.var = np.concatenate((fl1.p.reshape(data_sol.Nx,1)
                               ,pm1.K.reshape(data_sol.Nx,1)
                               ,fl1.mu.reshape(data_sol.Nx,1))
                              ,axis=1)

plot_out(data_sol) # call the plotting output


# Face only has one variable right now, so can directly plot
fig2, ax2 = plt.subplots()
ax2.plot(base_mesh.xc,fl1.u,color='black',linewidth=0.5)
ax2.set_xlabel('x (m)',fontsize=12) 
ax2.set_ylabel('u (m/s)',fontsize=12)
plt.xlim(min(base_mesh.xc),max(base_mesh.xc))
plt.ylim(min(fl1.u)-1E-6,max(fl1.u)+1E-6)
plt.show()
