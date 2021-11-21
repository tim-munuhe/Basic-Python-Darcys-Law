import multi_solv_v3 as ms
import darcy_v1 as dc
import scipy.sparse.linalg as SLA
from scipy.linalg import cho_factor, cho_solve
import csv
import numpy as np
from numpy import linalg as LA
import copy
import time

# Read Case
with open('casefile_mesh.csv',newline='') as casefile:  # open and read case params
    casereader = csv.DictReader(casefile) 
    i = 0 
    caselist = {} 
    for row in casereader:    
        caselist[i] = row 
        i += 1         

for i in range(0,len(caselist)):
      # Create case object and check values of variables
      case_current = dc.case_param(caselist[i])
      print(case_current.x0)
      print(case_current.dx)

      # Initialize and check mesh object creation ##
      case_mesh = dc.mesh(case_current) # create case mesh from case parameters
      Nx = case_mesh.Nx
      print('Node Locations w/ inlet:', case_mesh.x[0:5]) # check inlet location and spacing
      print('Nx:', case_mesh.Nx) # check number of elements
      print('Outlet Location:', case_mesh.x[case_mesh.Nx-1])
      print('Face Locations:', case_mesh.xc[0:5]) # 

      # Create fluid and porous medium objects for this specific case ##
      fl1 = dc.fluid(case_mesh,case_current.fl) # fluid object, determined by mesh and case's fluid properties
      pm1 = dc.por_med(case_mesh,case_current.pm) # porous medium object, determined by mesh and case's porous medium properties

      # Linear P
      print('Original P:',fl1.p[0:4],fl1.p[case_mesh.Nx-5:case_mesh.Nx-1])
      fl1.p_lin(case_mesh)
      print('Linear P:',fl1.p[0:4],fl1.p[case_mesh.Nx-5:case_mesh.Nx-1])

      # Darcy Velocity
      print('Original u:',fl1.u[0:4]) # velocity from initialization
      fl1.u = np.zeros(case_mesh.Nx) # zero out velocity 
      fl1.darcyv(case_mesh,pm1) # use darcyv method
      print('Darcy u:',fl1.u[0:4]) # print to confirm that darcyv did what it was supposed to (got same solution as initialization)

      ## Pressure calculation using Different Solvers

      # Gauss-Seidel Original
      fl_gso = dc.fluid(case_mesh,case_current.fl)
      print('Original P:',fl_gso.p[0:4],
            fl_gso.p[Nx-5:Nx-1])
      A,b = fl_gso.coeff_Ab(case_mesh,pm1)
      time_gso = time.perf_counter()
      [iter_gso, res_gso] = fl_gso.gauss_seidel(case_mesh,pm1,1.0E-9)
      time_gso -= time.perf_counter()
      time_gso = abs(time_gso)
      fl_gso.darcyv(case_mesh,pm1)

      # Gauss-Seidel
      fl_gs = dc.fluid(case_mesh,case_current.fl)
      # print('Original P:',fl_gs.p[0:4],
      #       fl_gs.p[Nx-5:Nx-1])
      A,b = fl_gs.coeff_Ab(case_mesh,pm1)
      time_gs = time.perf_counter()
      fl_gs.p, res_gs, iter_gs = ms.gauss_seidel(A,Nx,b,np.zeros(Nx),1.0E-9)
      time_gs -= time.perf_counter()
      time_gs = abs(time_gs)
      fl_gs.darcyv(case_mesh,pm1)

      # Gauss-Jordan
      fl_gj = dc.fluid(case_mesh,case_current.fl)
      time_gj = time.perf_counter()
      fl_gj.p, res_gj = ms.gauss_jordan(A,Nx,b)
      time_gj -= time.perf_counter()
      time_gj = abs(time_gj)
      fl_gj.darcyv(case_mesh,pm1)

      # Conjugate Gradient
      fl_cg = dc.fluid(case_mesh,case_current.fl)
      time_cg = time.perf_counter()
      fl_cg.p, info_cg = SLA.cg(A, b, tol=1.0E-9)
      time_cg -= time.perf_counter()
      time_cg = abs(time_cg)
      fl_cg.darcyv(case_mesh,pm1) 

      # Conjugate Gradient Squared
      fl_cgs = dc.fluid(case_mesh,case_current.fl)
      time_cgs = time.perf_counter()
      fl_cgs.p, info_cg = SLA.cgs(A, b, tol=1.0E-9)
      time_cgs -= time.perf_counter()
      time_cgs = abs(time_cgs)
      fl_cgs.darcyv(case_mesh,pm1) 

      # Cholesky Factorization
      fl_cho = dc.fluid(case_mesh,case_current.fl)
      time_cho = time.perf_counter()
      c_cho, low = cho_factor(A)
      fl_cho.p = cho_solve((c_cho, low), b)
      time_cho -= time.perf_counter()
      time_cho = abs(time_cho)
      fl_cho.darcyv(case_mesh,pm1)

      # Output

      # Data at nodes: x, K, mu, p_gs, p_gj, p_cg, p_cho
      data_case = type('sol', (object,), {})() # empty object (metaprogramming)
      data_case.Np = 2
      data_case.Nx = Nx
      data_case.varnamex = 'x (m)'
      data_case.varname = ['K ($m^2$)', '\u03BC (Pa*s)']
      data_case.fname = ['K_'+case_current.name+'.png','mu_'+case_current.name+'.png',]
      data_case.x = case_mesh.x
      data_case.var = np.zeros((data_case.Nx,data_case.Np))
      data_case.var = np.concatenate((pm1.K.reshape(data_case.Nx,1),
                                    fl_gs.mu.reshape(data_case.Nx,1))
                                    ,axis=1)
      dc.plot_out(data_case)

      data_sol = type('sol', (object,), {})() # empty object (metaprogramming)
      data_sol.Np = 4
      data_sol.Nx = case_mesh.Nx
      data_sol.varnamex = 'x (m)'
      data_sol.varname = ['p_gs (Pa)', 'p_gj (Pa)','p_cg (Pa)','p_cho (Pa)']
      data_sol.fname = 'p_'+case_current.name+'.png'

      data_sol.x = case_mesh.x
      data_sol.var = np.zeros((data_sol.Nx,data_sol.Np))
      data_sol.var = np.concatenate((fl_gs.p.reshape(data_sol.Nx,1),
                                    fl_gj.p.reshape(data_sol.Nx,1),
                                    fl_cg.p.reshape(data_sol.Nx,1),
                                    fl_cho.p.reshape(data_sol.Nx,1))
                                    ,axis=1)
      dc.multi_plot_out(data_sol,'p (Pa)')

      # Performance Reporting
      print('Timing (s): G-Seidel | G-Jordan | Conj. Grad | Cholesky')
      print('----------------------------------------------------------------')
      print(time_gs,'|',time_gj,'|',time_cg,'|',time_cho)
      print('Original Gauss-Seidel Timing:',time_gso)

      # Write timing and residual data to code
      data_write = type('sol', (object,), {})() # empty object (metaprogramming)
      data_write.name = case_current.name
      data_write.vartitle = ['Gauss-Seidel (Original)', 'Gauss-Seidel (Matrix)',
            'Gauss-Jordan','Conjugate Gradient','Cholesky']
      data_write.varname = ['Time (s)', 'Residual']
      data_write.var = [[time_gso,time_gs,time_gj,time_cg,time_cho],
            [LA.norm(fl1.p-fl_gso.p),LA.norm(fl1.p-fl_gs.p),
            LA.norm(fl1.p-fl_gj.p),LA.norm(fl1.p-fl_cg.p),
            LA.norm(fl1.p-fl_cho.p)]]
      dc.multi_write_out('Performance.csv',data_write)

# print('Gauss-Seidel P:',fl_gs.p[0:4],
#       fl_gs.p[case_mesh.Nx-5:case_mesh.Nx-1])
# fl_gs2 = copy.deepcopy(fl_gs) # another 100 iterations
# [itera, res] = fl_gs2.gauss_seidel(case_mesh,pm1)
# fl_gs3 = copy.deepcopy(fl_gs2) # another 100 iterations
# [itera, res] = fl_gs3.gauss_seidel(case_mesh,pm1)