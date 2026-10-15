#8:34 pm - 8:55 pm 
import numpy as np 



def findLU(A):
    U = A.copy()
    #U = U[:,:-1]
    #print(A.shape)
    #print(np.zeros(A.shape))
    L = np.array(np.zeros(A.shape))
    #L = L[:,:-1]
    for r in range(A.shape[1]):
        L[r][r]=1
    print(L)
    rows_num,_ = A.shape 
    for row in (range(rows_num)):
        pivot = U[row][row]
        for row_below in range(row+1,rows_num):
            multiplier = U[row_below][row]/pivot
            L[row_below][row] = multiplier 
            U[row_below] -= multiplier*U[row]
    return L,U



    
def backward_substitution(U,Z):
    solution_len = len(Z)
    sol = np.array(np.zeros(len(Z)))
    for i in range(solution_len-1,-1,-1):
        sol[i] = Z[i]
        for j in range(solution_len-1,i,-1):
            sol[i]-= U[i][j]*sol[j]
        sol[i] /= U[i][i]
    return sol


def backward_sub(U, Z):
    solution_len = len(Z)
    sol = np.array(np.zeros(solution_len))
    for i in range(solution_len-1, -1, -1):
        sol[i] = Z[i]
        
        for j in range(solution_len - 1, i, -1):
            sol[i] = sol[i] - U[i][j] * sol[j]
        sol[i] = sol[i] / U[i][i]
        
    return sol
            
        
        
        
        
    return sol 
def forward_substitution(L,C):
    #7:57 pm - 8:08 pm 
    solution_len = len(C)
    sol = np.array(np.zeros(len(C)))
    for i in range(solution_len):
        sol[i] = C[i] 
        for j in range(i-1,-1,-1):
            sol[i]-= L[i][j]*sol[j]
        sol[i] /= L[i][i]
    return sol  

def findInv(row_num,L,U):
    sol = np.array(np.zeros((row_num,row_num)))
    for i in range(row_num):
        iden_col = np.zeros((row_num))
        iden_col[i]=1
        Z = forward_substitution(L,iden_col)
        col = backward_substitution(U,Z)
        sol[i] = col 
    return sol.T 


def findInv(num_rows, L, U):
    sol = np.array(np.zeros((num_rows, num_rows)))

    for i in range(num_rows):
        # create the identity matrix
        iden_col = np.zeros((num_rows))
        iden_col[i] = 1

        Z = forward_substitution(L, iden_col)

        col = backward_substitution(U, Z)
        sol[i] = col
    # transpose the solution
    return sol.T
        
        
        
        


A = np.array([[10,2,3],[4,5,6],[7,8,9]],dtype='float64')
print(A)
print("-"*60)
L,U = findLU(A)
print(A)
print("-"*60)
print(L)
print("-"*60)
print(U)
print("-"*60)
print(L@U)
print("-"*60)
A_inv = findInv(A.shape[0],L,U)
print(A_inv)
print("-"*60)
print(A@A_inv)