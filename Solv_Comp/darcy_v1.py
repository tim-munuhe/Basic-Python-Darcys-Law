# Import packages
import csv # Read in case parameters and write out solutions
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import copy
from cycler import cycler

# Classes & Functions ----------------#
class case_param(): 
    def __init__(self,param): 
        self.name = param['case_name'] # now the name is given inside the case
                                       #, not as the case's actual name 
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
                self.x[i] = (self.xc[i+1] + self.xc[i])/2 # Cell Node Location
            self.x[self.Nx-1] = np.copy(self.xc[self.Nx-1]) # Outward boundary
    def output(self,fname): # output mesh
        with open(fname,'w', newline='') as csvfile:
            mesh_write = csv.writer(csvfile,dialect = 'excel', 
                                    delimiter = '\t') # writer object
            mesh_write.writerow(['i', 'x', 'xc']) # header row
            for i in range(0,self.Nx):
                mesh_write.writerow([i+1,self.x[i],self.xc[i]]) # actual data

class fluid():
    def __init__(self,mesh,fluid_prop):
        self.name = fluid_prop['Name']
        # Initialize variables
        self.p = np.ones(mesh.Nx)*fluid_prop['p0'] # Pressure
        self.p[mesh.Nx-1] = fluid_prop['pL'] # Pressure boundary at x = L
        self.u = np.ones(mesh.Nx)*fluid_prop['u0'] # Velocity: Staggered mesh 
        self.mu = np.ones(mesh.Nx)*fluid_prop['mu'] # Viscosity
    def p_lin(self,mesh): # linear pressure-x relation
        N = mesh.Nx
        L = mesh.x[N-1]
        L0 = mesh.x[0]
        for i in range(1,N):
            self.p[i] = (self.p[N-1]-self.p[0])/(L-L0)*mesh.x[i]+self.p[0]
    def mu_lin(self,mesh): # linear viscosity-x relation
        N = mesh.Nx
        L = mesh.x[N-1]
        L0 = mesh.x[0]
        for i in range(1,N):
            self.mu[i] = (0.005-0.001)/(L-L0)*mesh.x[i]+0.001
    def darcyv(self,msh,pm):
        # inlet
        self.u[0] = -np.mean([pm.K[0]/self.mu[0],pm.K[1]/self.mu[1]]) \
            *(self.p[1]-self.p[0])/(msh.x[1]-msh.x[0]) 
        self.u[1] = self.u[0] # same location
        for i in range(2,msh.Nx-1): # interior faces
            Ai = pm.K[i-1]/self.mu[i-1]/(msh.xc[i]-msh.x[i-1])
            Ai1 = pm.K[i]/self.mu[i]/(msh.x[i]-msh.xc[i])
            self.u[i] = -Ai*Ai1/(Ai+Ai1)*(self.p[i]-self.p[i-1])
        # outlet
        self.u[msh.Nx-1] = -np.mean([pm.K[msh.Nx-2]/self.mu[msh.Nx-2],        
                                    pm.K[msh.Nx-1]/self.mu[msh.Nx-1]])\
            *(self.p[msh.Nx-1]-self.p[msh.Nx-2])/(msh.x[msh.Nx-1]-msh.x[msh.Nx-2]) 
        
    def gauss_seidel(self,msh,pm,tol): # need the mesh info and pm permeability

        # Solver Parameters
        # tol = 1E-7 # tolerance to determining stopping point of scheme
        res = np.array([1.0],dtype=float) # residual (initially res > tol)
        max_iter = 1E5 # max iterations (so it doesn't go forever)
        k = 0 # iteration counter

        # self.p[2:N-1] = zeros(N-2,1) # initial guess for cell centers
        p_samp = np.zeros([1,4],dtype=float)
        p_samp[0][:] = np.copy([self.p[1],self.p[3],self.p[msh.Nx-4],
                               self.p[msh.Nx-2]])
        
        ## Iteration Loop
        while ((res[k]>tol)and(k<max_iter)):
            p_prev = np.copy(self.p)# previous iteration (not same mem loc)
            i = 1 # first cell center
            fw = -np.mean([pm.K[i-1]/self.mu[i-1],pm.K[i]/self.mu[i]])/(msh.x[i]-msh.xc[i]) # f_i-1/2 -> f_i
            fe = -pm.K[i]/self.mu[i]/(msh.xc[i+1]-msh.x[i]) # f_i -> f_i+1/2
            fee = -pm.K[i+1]/self.mu[i+1]/(msh.x[i+1]-msh.xc[i+1]) #f_i+1/2 -> f_i+1
            Aw = fw
            Ae = fe*fee/(fe+fee)
            self.p[i] = (Aw*self.p[i-1] + Ae*self.p[i+1])/(Aw + Ae)
            for i in range(2,msh.Nx-2):
                fww = -pm.K[i-1]/self.mu[i-1]/(msh.xc[i]-msh.x[i-1]) # f_i-1->i-1/2
                fw = -pm.K[i]/self.mu[i]/(msh.x[i]-msh.xc[i]) # f_i-1/2 -> f_i
                fe = -pm.K[i]/self.mu[i]/(msh.xc[i+1]-msh.x[i]) # f_i -> f_i+1/2
                fee = -pm.K[i+1]/self.mu[i+1]/(msh.x[i+1]-msh.xc[i+1]) # f_i+1/2 -> f_i+1
                Aw = fw*fww/(fw+fww) # "west" factor (i-1 -> i)
                Ae = fe*fee/(fe+fee) # "east" factor (i -> i+1)
                self.p[i] = (Aw*self.p[i-1] + Ae*self.p[i+1])/(Aw + Ae)
            i = msh.Nx-2 # last cell center
            fww = -pm.K[i-1]/self.mu[i-1]/(msh.xc[i]-msh.x[i-1]) # f_i-1->i-1/2
            fw = -pm.K[i]/self.mu[i]/(msh.x[i]-msh.xc[i]) # f_i-1/2 -> f_i
            fe = -np.mean([pm.K[i]/self.mu[i],pm.K[i+1]/self.mu[i+1]])/(msh.xc[i+1]-msh.x[i]) # f_i -> f_i+1/2
            Aw = fw*fww/(fw+fww) 
            Ae = fe 
            self.p[i] = (Aw*self.p[i-1] + Ae*self.p[i+1])/(Aw + Ae)
            p_samp = np.append(p_samp,[[self.p[1],self.p[3],self.p[msh.Nx-4]
                               ,self.p[msh.Nx-2]]],axis=0)
            res = np.append(res,[sum(abs(self.p-p_prev))]) # L2 norm of p_diff
            k += 1 # increase iteration count
            # print(k,res[k-1])
            
        # Iterations are complete. Now for output
        print('Gauss-Seidel Complete. Iteration, Residual:',k,res[k])
        
        # I suggest using the pandas library for output to file. Compare the 
        # code below to the output function coded from scratch in the mesh class
        res_vec = res[:,np.newaxis]
        df = pd.DataFrame(np.append(res_vec,p_samp,axis=1)
                          ,columns=['res','x1','x3','x_N-4','x_N-2'])
        df.to_csv('GS_Out.csv',sep='\t')       
        return [k,res[k]]

    def coeff_Ab(self,msh,pm):
        # Construct coefficient matrix A and source term b for Ax = b         
        # problem were x = P.
        N = msh.Nx
        A = np.eye(N,N,dtype=float)
        b = np.zeros(N)
        b[0] = self.p[0] # inlet BC
        b[N-1] = self.p[N-1] # outlet BC
        
        # Cell Center
        i = 1
        A[i,i-1] = -np.mean([pm.K[i-1]/self.mu[i-1],pm.K[i]/self.mu[i]])/(msh.x[i]-msh.xc[i]) # f_i-1/2 -> f_i
        fe = -pm.K[i]/self.mu[i]/(msh.xc[i+1]-msh.x[i]) # f_i -> 
        fee = -pm.K[i+1]/self.mu[i+1]/(msh.x[i+1]-msh.xc[i+1]) #f_i+1/2 -> f_i+1
        A[i,i+1] = fe*fee/(fe + fee)
        A[i,i] = -A[i,i+1] - A[i,i-1]
        for i in range(2,N-2): # loop through cells
            fww = -pm.K[i-1]/self.mu[i-1]/(msh.xc[i]-msh.x[i-1]) # f_i-1->i-1/2
            fw = -pm.K[i]/self.mu[i]/(msh.x[i]-msh.xc[i]) # f_i-1/2 -> f_i
            fe = -pm.K[i]/self.mu[i]/(msh.xc[i+1]-msh.x[i]) # f_i -> f_i+1/2
            fee = -pm.K[i+1]/self.mu[i+1]/(msh.x[i+1]-msh.xc[i+1]) # f_i+1/2 -> f_i+1
            A[i,i+1] = fw*fww/(fw+fww)
            A[i,i-1] = fe*fee/(fe+fee)
            A[i,i] = -A[i,i+1] - A[i,i-1]
        i = msh.Nx-2
        fww = -pm.K[i-1]/self.mu[i-1]/(msh.xc[i]-msh.x[i-1]) # f_i-1->i-1/2
        fw = -pm.K[i]/self.mu[i]/(msh.x[i]-msh.xc[i]) # f_i-1/2 -> f_i
        A[i,i-1] = fw*fww/(fw + fww)
        A[i,i+1] = -np.mean([pm.K[i]/self.mu[i],pm.K[i+1]/self.mu[i+1]])/(msh.xc[i+1]-msh.x[i]) # f_i -> f_i+1/2
        A[i,i] = -A[i,i-1] - A[i,i+1]

        return A, b   

class por_med():
    def __init__(self,mesh,pm_prop):
        self.name = pm_prop
        # Initialize Variables
        self.K = np.ones(mesh.Nx)*pm_prop['K'] # Permeability
        self.eps = np.ones(mesh.Nx)*pm_prop['eps'] # Porosity


def plot_out(data): # plotting function. in: data object. out: plot
    N_param = data.Np
    
    N = data.Nx
    for i in range(0,N_param):
        fig, ax = plt.subplots(figsize=(4,5))   
        ax.plot(data.x, data.var[0:,i], linewidth=0.5)
        ax.set_xlabel(data.varnamex,fontsize=12) 
        ax.set_ylabel(data.varname[i],fontsize=12)
        fig.tight_layout()
        plt.xlim(min(data.x),max(data.x))
        plt.savefig(data.fname[i])

    # plt.show()

    return None

def multi_plot_out(data,ylab): # plotting function. in: data object. out: plot
    N_param = data.Np
    
    N = data.Nx
    cust_cycler = (cycler(color=['r', 'g', 'b', 'k']) +
                  cycler(linestyle=['-', '--', ':', '-.']))
    fig, ax = plt.subplots(figsize=(4,5))   
    ax.set_prop_cycle(cust_cycler) 
    ax.set_xlabel(data.varnamex,fontsize=12) 
    ax.set_ylabel(ylab,fontsize=12)
    for i in range(0,N_param):
        ax.plot(data.x, data.var[0:,i], linewidth=0.5, label=data.varname[i])
        ax.legend()
        fig.tight_layout()

    plt.xlim(min(data.x),max(data.x))    
    # plt.ylim(min(data.var.flatten()),max(data.var.flatten()))
    # plt.show()
    plt.savefig(data.fname)

    return None


def multi_write_out(fname,data): # writing function. in: data object. out: csv
    with open(fname,'a',newline='') as csvfile:
        data_writer = csv.writer(csvfile, delimiter=',')
        row_1 = []
        row_1.append(data.name)
        for i in range(0,len(data.vartitle)): row_1.append(data.vartitle[i])
        data_writer.writerow(row_1)
        for i in range(0,len(data.varname)):
            row_2 = []
            row_2.append(data.varname[i])
            for j in range(0,len(data.var[i][0:])): row_2.append(data.var[i][j])
            data_writer.writerow(row_2)
    
    return None