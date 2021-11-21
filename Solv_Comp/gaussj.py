# Linear eq soln by Gauss-Jordan elimination. A is NxN matrix
# stored in NPxNP array. B is an input matrix of NxM containing
# the M right-hand side vectors stored in NPxMP array. Output
# A^-1 is A inverse, stored in A, solution to AX = B is stored in B 
import numpy as np
from numpy import linalg as LA
import csv # Write solutions to file

def gaussj_NR(A,N,B,M):
    # Ripped from FORTRAN Numerical Recipes Textbook
    # Gauss-Jordan Elimination w/ Full Pivoting (2.1)
    
    # Initialize index arrays
    # nmax = 50
    ipiv, indxr, indxc  = np.zeros((3,N),dtype=int) # initialize indices
    # Main column reduction loop
    for i in range(0,N): 
        big = 0.0
        for j in range(0,N): # Pivot element search
            if (ipiv[j] != 1):
                for k in range(0,N):
                    if (ipiv[k] == 0):
                        if (abs(A[j,k])>=big):
                            big = abs(A[j,k])
                            irow = j
                            icol = k
                    elif (ipiv[k]>1): raise Exception('Singular Matrix')
        ipiv[icol] += 1 
        # Permute Matrix to put pivot element on diagonal
        if (irow.ne.icol):
            for l in range(1,N):
                dum = A[irow,l]
                A[irow,l] = A[icol,l]
                A[icol,l] = dum
            for l in range(1,M):
                dum = B[irow,l]
                B[irow,l] = B[icol,l]
                B[icol,l] = dum
        # Divide pivot row by pivot element
        indxr[i] = irow
        indxc[i] = icol
        if (A[icol,icol] == 0.0):
            raise Exception('Singular Matrix')
        pivinv = 1.0/A[icol,icol]
        A[icol,icol] = 1.0
        for l in range(1,N): A[icol,l] = A[icol,l]*pivinv
        for l in range(1,M): B[icol,l] = B[icol,l]*pivinv
        # Row Reduction
        for ll in range(1,N): 
            if (ll != icol):
                dum = A[ll,icol]
                A[ll,icol] = 0.0
                for l in range(0,N): A[ll,l] -= A[icol,l]*dum
                for l in range(0,M): B[ll,l] -= B[icol,l]*dum
    # Reverse permutation
    for l in range(N-1, -1, -1):
        if (indxr[l] != indxc[l]):
            for k in range(0,N):
                dum = A[k,indxr[l]]
                A[k,indxr[l]] = A[k,indxc[l]]
                A[k,indxc[l]] = dum
    return A, B
    
def gaussj_cust(A,N,b):
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
            P = np.eye(N) # Permutation matrix for this column pivot
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
    print_sol('sol',A_org,A_inv,b,np.matmul(A_inv,b_org),N)

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

def print_mat_single(i, A, N):
    # Print the A matrix in formatted form (CSV file). N is dimensional length
    
    fname = 'mat_{}.csv'.format(i) # file name
    print('Single Matrix: ',i)
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
rng = np.random.default_rng(1235)
A = rng.integers(low=-10, high=10, size=(N,N)).astype(np.float)
x = rng.integers(low=-10, high=10, size=N).astype(np.float)
b = np.matmul(A,x)
print_mat_single('A_Original',A,N)
print_mat3('Original',A,np.eye(N),np.eye(N),b,N)

# Run and output
# [A_inv_gj, x_gj] = gaussj(A, N, b, 1)
[x_gj, A_inv_gj] = gaussj_cust(A, N, b)
print('A_inv shape:',np.shape(A_inv_gj))
print('x shape:',np.shape(x_gj.shape))
print_mat3('Final',A,A_inv_gj,np.eye(N),x_gj,N)