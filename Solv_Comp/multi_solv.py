# Linear eq soln by Gauss-Jordan elimination. A is NxN matrix
# stored in NPxNP array. B is an input matrix of NxM containing
# the M right-hand side vectors stored in NPxMP array. Output
# A^-1 is A inverse, stored in A, solution to AX = B is stored in B 
import numpy as np
from numpy import linalg as LA
from scipy.sparse.linalg import cg
from scipy.linalg import cho_factor, cho_solve
import csv

def gaussj_cust(A,N,b):
    # Custom Gauss-Jordan Elimination Algo
    # Notes from 9/11-13 2021 (Math folder)
    A_org = np.copy(A) # Original A (need for tests)
    b_org = np.copy(b)
    A_inv = np.eye(N) # A inverse initialized as identity matrix
    P_gen = np.eye(N) # General Permutation matrix
    P = np.eye(N) # Permutation matrix for this column pivot
    point_name = 'org_func'
    print_mat3(point_name, A, A_inv, P_gen, b, N)
    # Reduce to upper tridiagonal coefficient matrix
    for i in range(0, N-1):
        # Find row of pivot element in column i
        # i_max = np.where(A[i:,i] == np.amax(A[i:,i]))
        i_max = i
        for j in range(i+1,N):
          if (abs(A[i_max,i]) < abs(A[j,i])): i_max = j
        if (A[i_max,i] < 0.0):
            A[i_max,0:] = A[i_max,0:]*-1.0 # Switch signs
            A_inv[i_max,0:] = A_inv[i_max,0:]*-1.0
            b[i_max] = b[i_max]*-1.0
        # Move pivot row so column is on diagonal
        if (i_max != i): # Swap rows
            P[i_max, i_max] = 0.0
            P[i, i] = 0.0
            P[i_max, i] = 1.0 # Move row i to row i_max
            P[i, i_max] = 1.0 # Move row i_max to row i
            A = np.matmul(P,A)
            A_inv = np.matmul(P,A_inv) 
            b = np.matmul(P,b)
            P_gen = np.matmul(P, P_gen) # !compt. intensive!
            # Debugging output
            point_name = 'Swap_{}'.format(i_max)
            print_mat3(point_name, A, A_inv, P_gen, b, N)
        # Pivot
        fac = A[i,i]
        b[i] = b[i]/fac # A[i,i] is now pivot element
        A_inv[i,0:] = A_inv[i,0:]/fac
        A[i,0:] = A[i,0:]/fac # pivot element becomes 1
        # Debugging output
        point_name = 'Pivot_{}'.format(i)
        print_mat3(point_name, A, A_inv, P_gen, b, N)
        # Cancel non-zero column elements below pivot row
        for j in range(i+1,N): # Other rows
            if  (A[j,i] != 0.0): # skip rows with column element = 0
                fac = A[j,i] # factor for multiplying pivot row
                A[j,0:] -= fac*A[i,0:]
                A_inv[j,0:] -= fac*A_inv[i,0:]
                b[j] -= fac*b[i]
                # Debugging output
                point_name = 'Cancelb_{}'.format(i)
                print_mat3(point_name, A, A_inv, P_gen, b, N)
    if (A[N-1,N-1] != 0.0):
        fac = A[N-1,N-1]
        b[N-1] = b[N-1]/fac # last row
        A_inv[N-1,0:] = A_inv[N-1,0:]/fac
        A[N-1,0:] = A[N-1,0:]/fac
        # Debugging output
        point_name = 'Last Row_{}'.format(i)
        print_mat3(point_name, A, A_inv, P_gen, b, N)
    elif (b[N-1] != 0.0): # No solution here
        raise Exception('Singular matrix: 0*x_N != 0.0')
    
    # Check: Matrix is in upper tridiagonal form
    if (LA.norm(A-np.triu(A)) < 1.0E-14):
        print('>> A reduced to upper tridiagonal matrix! <<')
    else: 
        print('> Partial Reduction Failed!')
        return x, A
    
    # Back sub for trailing terms (full pivoting)
    for i in range(N-1,0,-1): # Start: bottom row cancels out other rows
        for j in range(i-1,-1,-1): # Other rows
            if (A[j,i] != 0.0):
                fac = A[j,i]
                A[j,0:] -= fac*A[i,0:]
                A_inv[j,0:] -= fac*A_inv[i,0:]
                b[j] -= fac*b[i]
                # Debugging output
                point_name = 'Back_Sub_{}_{}'.format(i,j)
                print_mat3(point_name, A, A_inv, P_gen, b, N)
        # Debugging output
        point_name = 'Back_Sub_i_{}'.format(i)
        print_mat3(point_name, A, A_inv, P_gen, b, N)
    
    # Revert solutions back to original order
    P_trans = np.transpose(P)
    A_inv_perm = np.matmul(P_trans,A_inv)
    
    ## Test Solution

    # Identity test: A_inv x A = I
    I_test = np.matmul(A_inv,A_org)
    print_mat3('I_test', A_org, A_inv, I_test, b, N)
    I_test_perm = np.matmul(A_inv_perm,A_org)
    print_mat3('I_test_perm', A_org, A_inv, I_test_perm, b, N)    
    
    # Solution Test: norm(x - (A_inv x b)) < some precision number
    l2_res = LA.norm(b_org-np.matmul(A_inv,b_org))
    print('Gauss-Jordan: Residual = ',l2_res)
    print_sol('GJ_sol',A_org,A_inv,b,np.matmul(A_inv,b_org),N)

    # Check: Full pivoting has turned A into identity matrix
    if (LA.norm(A-np.eye(N))<1.0E-14): 
        print('>> Full Pivoting Complete! <<')
        return x, A_inv
    else:
        print('> Full Pivoting Failed!')
        return x, A

def gaussj_cust_noSwap(A,N,b):
    # Custom Gauss-Jordan Elimination Algo
    # Notes from 9/11-13 2021 (Math folder)
    A_org = np.copy(A) # Original A (need for tests)
    b_org = np.copy(b)
    A_inv = np.eye(N) # A inverse initialized as identity matrix
    P_gen = np.eye(N) # General Permutation matrix
    point_name = 'org_func'
    print_mat3(point_name, A, A_inv, P_gen, b, N)
    # Reduce to upper tridiagonal coefficient matrix
    for i in range(0, N-1):
        # Pivot
        fac = A[i,i]
        b[i] = b[i]/fac # A[i,i] is now pivot element
        A_inv[i,i:] = A_inv[i,i:]/fac
        A[i,i:] = A[i,i:]/fac # pivot element becomes 1
        # Debugging output
        point_name = 'Pivot_{}'.format(i)
        print_mat3(point_name, A, A_inv, P_gen, b, N)
        # Cancel non-zero column elements below pivot row
        for j in range(i+1,N): 
            if  (A[j,i] != 0.0): # skip rows with column element = 0
                fac = A[j,i] # factor for multiplying pivot row
                for k in range(i,N): # column iteration
                    A[j,k] -= fac*A[i,k]
                    A_inv[j,k] -= fac*A_inv[i,k]
                    # Debugging output
                    point_name = 'Cancel_{}_{}'.format(i,k)
                    print_mat3(point_name, A, A_inv, P_gen, b, N)
                b[j] -= fac*b[i]
                # Debugging output
                point_name = 'Cancelb_{}'.format(i)
                print_mat3(point_name, A, A_inv, P_gen, b, N)
    if (A[N-1,N-1] != 0.0):
        fac = A[N-1,N-1]
        b[N-1] = b[N-1]/fac # last row
        A_inv[N-1] = A_inv[N-1]/fac
        A[N-1,N-1] = A[N-1,N-1]/fac
        # Debugging output
        point_name = 'Last Row_{}'.format(i)
        print_mat3(point_name, A, A_inv, P_gen, b, N)
    elif (b[N-1] != 0.0): # No solution here
        raise Exception('Singular matrix: 0*x_N != 0.0')
    
    # Check: Matrix is in upper tridiagonal form
    if (LA.norm(A-np.triu(A)) < 1.0E-14):
        print('>> A reduced to upper tridiagonal matrix! <<')
    else: 
        print('> Partial Reduction Failed!')
        return x, A
    
    # Back sub for trailing terms (full pivoting)
    for i in range(N-1,0,-1): # Start: bottom row cancels out other rows
        for j in range(i-1,-1,-1): # Other rows
            if (A[j,i] != 0.0):
                fac = A[j,i]
                A[j,0:] -= fac*A[i,0:]
                A_inv[j,0:] -= fac*A_inv[i,0:]
                b[j] -= fac*b[i]
                # Debugging output
                point_name = 'Back_Sub_{}_{}'.format(i,j)
                print_mat3(point_name, A, A_inv, P_gen, b, N)
        # Debugging output
        point_name = 'Back_Sub_i_{}'.format(i)
        print_mat3(point_name, A, A_inv, P_gen, b, N)
    
    # Revert solutions back to original order
    P_trans = np.transpose(P_gen)
    A_inv_perm = np.matmul(P_trans,A_inv)
    
    ## Test Solution

    # Identity test: A_inv x A = I
    I_test = np.matmul(A_inv,A_org)
    print_mat3('I_test', A_org, A_inv, I_test, b, N)
    I_test_perm = np.matmul(A_inv_perm,A_org)
    print_mat3('I_test_perm', A_org, A_inv, I_test_perm, b, N)    
    
    # Solution Test: norm(x - (A_inv x b)) < some precision number
    l2_res = LA.norm(b_org-np.matmul(A_inv,b_org))
    print('Gauss-Jordan: Residual = ',l2_res)
    print_sol('sol',A_org,A_inv,b,np.matmul(A_inv,b_org),N)

    # Check: Full pivoting has turned A into identity matrix
    if (LA.norm(A-np.eye(N))<1.0E-14): 
        print('>> Full Pivoting Complete! <<')
        return x, A_inv
    else:
        print('> Full Pivoting Failed!')
        return x, A

def gauss_seidel_cust(A,N,b,x0):
    # Gauss-Seidel: x0 is the initial solution
    # No inverse matrix to return, just iteration states
    ## Initialize
    x = x0
    dif = np.matmul(A,x)-b
    res = np.array([LA.norm(dif)],dtype=float)
    tol = 1E-8
    iter = 0
    max_iter = 100
    print('> Gauss-Seidel Initial Residual: {}'.format(res[0]))
    ## Iteration Loop
    while ((res[iter] > tol) & (iter < max_iter)):
        iter += 1
        for i in range(0,N): # Loop through equations
            x_sum = b[i]
            for j in range(0,N): 
                if (i != j): x_sum -= A[i,j]*x[j]
            x[i] = x_sum/A[i,i]
        dif = np.matmul(A,x)-b
        res_iter = np.array(LA.norm(dif))
        res = np.append(res,res_iter) # check this code
        print('> Gauss-Seidel Residual at iter {}: {}'.format(iter,res_iter))

    if (res[iter] < tol): print('>> Gauss-Seidel Converged! tol = ',tol,', res = ',res[iter])
    if (res[iter] > tol): print('> Gauss-Seidel Divergence! tol = ',tol,', res = ',res[iter])
    print_sol('Gauss-Seidel_Sol',A,A,x,np.matmul(A,x),N)

    return x, res

def print_mat3(i, A, A_inv, P, b, N):
    # Print the A, inverse A, permutation P matrices, and b vector 
    # in formatted form (CSV file). N is length of b-vector and dimensional length of every matrix
    
    fname = 'mat_{}.csv'.format(i) # file name
    # Create writer object
    with open(fname, 'w', newline = '') as f:
        mat_writer = csv.writer(f, delimiter = ',')
        for j in range(0,N):
            mat_writer.writerow(np.concatenate((A[j,0:],np.append(A_inv[j,0:],b[j]),P[j,0:])))
     
    return None

def print_mat1(i, A, N):
    # Print the NxN-sized A matrix in formatted form (CSV file). 

    fname = 'mat_{}.csv'.format(i) # file name
    # Create writer object
    with open(fname, 'w', newline = '') as f:
        mat_writer = csv.writer(f, delimiter = ',')
        for j in range(0,N):
            mat_writer.writerow(A[j,0:])
     
    return None

def print_sol(i, A, A_inv, x, b, N):
    # Print the A and inverse A matrices and x and b vectors in formatted
    # form (CSV file). N is length of b-vector and dimensional length of every matrix
    
    fname = 'mat_{}.csv'.format(i) # file name
    # Create writer object
    with open(fname, 'w', newline = '') as f:
        mat_writer = csv.writer(f, delimiter = ',')
        for j in range(0,N):
            mat_writer.writerow(np.concatenate((A[j,0:],A_inv[j,0:],[x[j],b[j]])))
     
    return None

    
# Test: Create random A (10*10) and x (10*1) arrays, multiply to get b, then solve for x using A and b. Simple

# Random A and x generation
N = 7
rng = np.random.default_rng(124)
# A: symmetric positive definite
A_seed = rng.integers(low=1, high=6, size=(N,N)).astype(np.float)
A_seed_T = np.transpose(A_seed)
A = (A_seed + A_seed_T)*0.5
A += np.diag(np.ones(N)*10.0)
A_org = A
print('Determinant of A:',LA.det(A))
x = rng.integers(low=-10, high=10, size=N).astype(np.float)
b = np.matmul(A,x)
print_mat1('A_Org',A,N)
print_mat3('Original',A,np.eye(N),np.eye(N),b,N)

# Gauss-Seidel
[x_gs, res_gs] = gauss_seidel_cust(A, N, b, np.ones(N,dtype=float)*-0.1)
print_mat3('GS_Final',A,A_org,np.eye(N),x_gs,N)

# Gauss-Jordan
# [A_inv_gj, x_gj] = gaussj(A, N, b, 1)
[x_gj, A_inv_gj] = gaussj_cust(A, N, b)
print('A_inv shape:',np.shape(A_inv_gj))
print('x shape:',np.shape(x_gj.shape))
print_mat3('GJ_Final',A,A_inv_gj,np.eye(N),x_gj,N)

# Conjugate Gradient
[x_cg, info_cg] = cg(A_org, b) 
print_mat3('CG_Final',A,A_org,np.eye(N),x_cg,N)

# Cholesky Factorization
c_cho, low = cho_factor(A_org)
x_cho = cho_solve((c_cho, low), b)
print_mat3('Cho_Final',A,A_org,np.eye(N),x_cho,N)