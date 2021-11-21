# Linear eq soln by Gauss-Jordan elimination. A is NxN matrix
# stored in NPxNP array. B is an input matrix of NxM containing
# the M right-hand side vectors stored in NPxMP array. Output
# A^-1 is A inverse, stored in A, solution to AX = B is stored in B 
import numpy as np
from numpy import linalg as LA
from scipy.sparse.linalg import cg
from scipy.linalg import cho_factor, cho_solve
import csv

def gauss_jordan(A,N,b):
    # Custom Gauss-Jordan Elimination Algo

    # Notes from 9/11-13 2021 (Math folder)
    A_org = np.copy(A) # Original A (need for tests)
    b_org = np.copy(b)
    A_inv = np.eye(N) # A inverse initialized as identity matrix
    P_gen = np.eye(N) # General Permutation matrix
    P = np.eye(N) # Permutation matrix for this column pivot
    # Reduce to upper tridiagonal coefficient matrix
    for i in range(0, N-1):
        # Find row of pivot element in column i
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
        # Pivot
        fac = A[i,i]
        b[i] = b[i]/fac # A[i,i] is now pivot element
        A_inv[i,0:] = A_inv[i,0:]/fac
        A[i,0:] = A[i,0:]/fac # pivot element becomes 1
        for j in range(i+1,N): # Other rows
            if  (A[j,i] != 0.0): # skip rows with column element = 0
                fac = A[j,i] # factor for multiplying pivot row
                A[j,0:] -= fac*A[i,0:]
                A_inv[j,0:] -= fac*A_inv[i,0:]
                b[j] -= fac*b[i]
    if (A[N-1,N-1] != 0.0):
        fac = A[N-1,N-1]
        b[N-1] = b[N-1]/fac # last row
        A_inv[N-1,0:] = A_inv[N-1,0:]/fac
        A[N-1,0:] = A[N-1,0:]/fac
    elif (b[N-1] != 0.0): # No solution here
        raise Exception('Singular matrix: 0*x_N != 0.0')
    
    # Check: Matrix is in upper tridiagonal form
    if (LA.norm(A-np.triu(A)) <= 1.0E-14):
        print('>> A reduced to upper tridiagonal matrix! <<')
    else: 
        print('> Partial Reduction Failed!')
        return b, A
    
    # Back sub for trailing terms (full pivoting)
    for i in range(N-1,0,-1): # Start: bottom row cancels out other rows
        for j in range(i-1,-1,-1): # Other rows
            if (A[j,i] != 0.0):
                fac = A[j,i]
                A[j,0:] -= fac*A[i,0:]
                A_inv[j,0:] -= fac*A_inv[i,0:]
                b[j] -= fac*b[i]
    
    # Revert solutions back to original order
    P_trans = np.transpose(P)
    A_inv_perm = np.matmul(P_trans,A_inv)
   
    # Solution Test: norm(x - (A_inv x b)) < some precision number
    l2_res = LA.norm(b-np.matmul(A_inv,b_org))

    # Check: Full pivoting has turned A into identity matrix
    if (LA.norm(A-np.eye(N))<1.0E-14): 
        return b, A_inv
    else:
        print('> Full Pivoting Failed!')
        return b, A

def gauss_seidel(A,N,b,x0,tol):
    # Gauss-Seidel: x0 is the initial solution
    # No inverse matrix to return, just iteration states

    ## Initialize
    x = np.copy(x0)
    dif = np.matmul(A,x)-b
    res = np.array([LA.norm(dif)],dtype=float)
    if (N < 10):
        # x_out = np.copy(x0).T # tracking solution
        x_out = np.reshape(x0,(N,1))
    # tol = 1E-8
    iter = 0
    max_iter = 1E5
    # print('> Gauss-Seidel Initial Residual: {}'.format(res[0]))
    
    # Output
    varname = []
    NVar = 0
    
    ## Iteration Loop
    res_len = len(res)
    while ((res[res_len-1] > tol) & (iter < max_iter)):
        iter += 1
        for i in range(0,N): # Loop through equations
            x_sum = b[i]
            for j in range(0,N): 
                if (i != j): x_sum -= A[i,j]*x[j]
            x[i] = x_sum/A[i,i]

        if ((iter % 10) == 0):
            dif = np.matmul(A,x)-b
            res_iter = np.array(LA.norm(dif))
            res = np.append(res,res_iter) # check this code
            res_len = len(res)
        # print('> Gauss-Seidel Residual at iter {}: {}'.format(iter,res_iter))

        if (N < 10): 
            x_out = np.hstack((x_out,np.reshape(x,(N,1))))

    if (res[res_len-1] <= tol): 
        print('>> Gauss-Seidel Converged at iteration',iter,\
            '! tol =',tol,', res =',res[res_len-1])
    else:
        print('> Gauss-Seidel Divergence at iteration',iter,\
            '! tol =',tol,', res =',res[res_len-1])
    print_sol('Gauss-Seidel_Sol',A,A,x,np.matmul(A,x),N)

    # Output if not too many elements
    if (N < 10):
        print_mat4('GS_Track',3,['A','x','b'],A,x_out,b)
        print('x shape, N, iter:',x_out.shape,N, iter)
        fname = 'GS_Sol_Track.csv'.format(i) # file name
        with open(fname, 'w', newline = '') as f:
            mat_writer = csv.writer(f, delimiter = ',')
            for j in range(0,N):
                mat_writer.writerow(x_out[j,0:])

    return x, res[res_len-1], iter

def print_mat3(i, A, A_inv, P, b, N):
    # Print the A, inverse A, permutation P matrices, and b vector 
    # in formatted form (CSV file). N is length of b-vector and dimensional length of every matrix
    
    fname = 'mat_{}.csv'.format(i) # file name
    f_header = 'A'*N + 'a'*N + 'b' + 'P'*N
    # Create writer object
    with open(fname, 'w', newline = '') as f:
        mat_writer = csv.writer(f, delimiter = ',')
        mat_writer.writerow(f_header)
        for j in range(0,N):
            mat_writer.writerow(np.concatenate((A[j,0:],np.append(A_inv[j,0:],b[j]),P[j,0:])))
     
    return None

def print_mat4(iname,NVar,varname,*args):
    # Print the A, inverse A, permutation P matrices, and b vector 
    # in formatted form (CSV file). N is length of b-vector and dimensional length of every matrix
    # Input
    # iname: File name
    # NVar: Number of variables to print
    # varname: Variable names given in array.
    # *args: matrices/vectors to be printed to csv, in order given by varname 
    # Must be same number of variable names and args as NVar 

    # Check & Unpack
    if (len(args) != NVar): raise Exception('print_mat4: Wrong number of variables!')
    if (len(varname) != NVar): raise Exception('print_mat4: NVar != # of Variable Names!')
    N = len(args[0])
    Ncol = np.zeros(NVar, dtype=int) # number of columns
    mat_out = args[0] # initialize
    Ncol[0] = args[0].shape[1]
    for j in range(1,NVar):
        if ((args[j].size % N) == 0): # check array sizes are compatible
            if (args[j].ndim == 1): 
                mat_out = np.hstack((mat_out,args[j].reshape(N,1)))
                Ncol[j] = 1
            elif (args[j].ndim == 2):
                if (args[j].shape[0] == N):
                    mat_out = np.hstack((mat_out,args[j]))
                    Ncol[j] = args[j].shape[1]
                elif (args[j].shape[1] == N): # assuming just transpose
                    mat_out = np.hstack((mat_out,np.transpose(args[j])))
                    Ncol[j] = np.transpose(args[j]).shape[1]
                    print('print_mat4: Check argument transpose, N = ',N)
                else:
                    mat_out = np.hstack((mat_out,\
                        args[j].reshape(N,int(args[j].size/N))))
                    Ncol[j] = args[j].reshape(N,int(args[j].size/N)).shape
                    print('print_mat4: Check argument shape, N = ',N)
        else:
            raise Exception('print_mat4: Argument size is not correct:',\
                N,args[j].shape)

    # Output
    fname = 'mat_{}.csv'.format(iname) # file name
    f_header = np.empty(0,dtype=str) # Figure this out
    for j in range(0,NVar):
        for i in range(0,Ncol[j]):
            f_header = np.append(f_header,varname[j])
    # header must know NVar and NCol for each variable
    # Create writer object
    with open(fname, 'w', newline = '') as f:
        mat_writer = csv.writer(f, delimiter = ',')
        mat_writer.writerow(f_header)
        for j in range(0,N):
            mat_writer.writerow(mat_out[j,0:])     
    return None


def print_mat1(i, A, N):
    # Print the NxN-sized A matrix in formatted form (CSV file). 
    # i: designation (number or name)
    # N: size of matrix
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
            mat_writer.writerow(np.concatenate(\
                (A[j,0:],A_inv[j,0:],[x[j],b[j]])))
     
    return None

def rand_SPD_Axb(N,gen_num):
    # Random A, x, and corresponding generation for Ax = b problem
    rng = np.random.default_rng(gen_num)
    # A: symmetric positive definite
    A_seed = rng.integers(low=-5, high=5, size=(N,N)).astype(np.float)
    A_seed_T = np.transpose(A_seed)
    A = (np.abs(A_seed) + np.abs(A_seed_T))*0.5
    A += np.diag(np.ones(N)*N)
    A_org = A
    print('Determinant of A:',LA.det(A))
    x = rng.integers(low=-N, high=N, size=N).astype(np.float)
    b = np.matmul(A,x)
    print_mat1('A_Org',A,N)
    print_mat3('Original',A,np.eye(N),np.eye(N),b,N)
    
    return A, x, b

def row_reduc(A_org):
    N,M = A_org.shape
    if (N != M): raise Exception('Matrix must be square! Size:',N,',',M)
    
    # Setup
    A = np.copy(A_org)
    A_inv = np.eye(N) # A inverse initialized as identity matrix
    P_gen = np.eye(N) # General Permutation matrix
    P = np.eye(N) # Permutation matrix for this column pivot

    # Reduce to upper tridiagonal coefficient matrix
    for i in range(0, N-1):
        # Find row of pivot element in column i
        i_max = i
        for j in range(i+1,N):
          if (abs(A[i_max,i]) < abs(A[j,i])): i_max = j
        if (A[i_max,i] < 0.0):
            A[i_max,0:] = A[i_max,0:]*-1.0 # Switch signs
            A_inv[i_max,0:] = A_inv[i_max,0:]*-1.0
        # Move pivot row so column is on diagonal
        if (i_max != i): # Swap rows
            P[i_max, i_max] = 0.0
            P[i, i] = 0.0
            P[i_max, i] = 1.0 # Move row i to row i_max
            P[i, i_max] = 1.0 # Move row i_max to row i
            A = np.matmul(P,A)
            A_inv = np.matmul(P,A_inv) 
            P_gen = np.matmul(P, P_gen) # !compt. intensive!
        # Pivot
        fac = A[i,i]
        A_inv[i,0:] = A_inv[i,0:]/fac
        A[i,0:] = A[i,0:]/fac # pivot element becomes 1
        for j in range(i+1,N): # Other rows
            if  (A[j,i] != 0.0): # skip rows with column element = 0
                fac = A[j,i] # factor for multiplying pivot row
                A[j,0:] -= fac*A[i,0:]
                A_inv[j,0:] -= fac*A_inv[i,0:]
    if (A[N-1,N-1] != 0.0):
        fac = A[N-1,N-1]
        A_inv[N-1,0:] = A_inv[N-1,0:]/fac
        A[N-1,0:] = A[N-1,0:]/fac                
    return A 

# # Test: Create random A (10*10) and x (10*1) arrays, multiply to get b, then solve for x using A and b.
# rng2 = np.random.default_rng()
# N = rng2.integers(6,14)
# gen_num = rng2.integers(10,100000)
# A, x, b = rand_SPD_Axb(N, gen_num)
# A_org, x_org = np.copy(A), np.copy(x)

# A_red = row_reduc(A)
# print_mat1('rand_red',A_red,N)

# # Gauss-Seidel
# [x_gs, res_gs, iter_gs] = gauss_seidel(A, N, b, np.ones(N,dtype=float)*0.1,1E-8)
# print_mat3('GS_Final',A,A_org,np.diag(x),x_gs,N)
# print_mat4('GS_Final2',4,['A','b','x_true','x_gs'],A,b,x,x_gs)

# # Gauss-Jordan
# [x_gj, A_inv_gj] = gauss_jordan(A, N, b)
# print('A_inv shape:',np.shape(A_inv_gj))
# print('x shape:',np.shape(x_gj.shape))
# print_mat3('GJ_Final',A,A_inv_gj,np.diag(x),x_gj,N)

# # Conjugate Gradient
# [x_cg, info_cg] = cg(A_org, b) 
# print_mat3('CG_Final',A,A_org,np.diag(x),x_cg,N)

# # Cholesky Factorization
# c_cho, low = cho_factor(A_org)
# x_cho = cho_solve((c_cho, low), b)
# print_mat3('Cho_Final',A,A_org,np.diag(x),x_cho,N)

# # Result Reporting
# print('Residuals: G-Seidel | G-Jordan | Conj. Grad | Cholesky')
# print('----------------------------------------------------------------')
# print(LA.norm(x_org-x_gs),' | ',LA.norm(x_org-x_gj),' | '\
#     ,LA.norm(x_org-x_cg),' | ',LA.norm(x_org-x_cho))