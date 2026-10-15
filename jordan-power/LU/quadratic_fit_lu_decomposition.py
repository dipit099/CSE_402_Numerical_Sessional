#7:22 pm - 8:28 pm 
import numpy as np 

def defineA(t_list,v_list):
    A = np.array(np.zeros((len(t_list),4)),dtype='float64')
    print(A)
    for i in range(len(t_list)):
        A[i][0] = t_list[i]**2 
        A[i][1] = t_list[i]
        A[i][2] = 1
        A[i][3] = v_list[i]
    return A 


def findLU(A):
    U = A.copy()
    U = U[:,:-1]
    print(A.shape)
    print(np.zeros(A.shape))
    L = np.array(np.zeros(A.shape))
    L = L[:,:-1]
    for r in range(A.shape[1]-1):
        L[r][r]=1
    print(L)
    rows_num,_ = A.shape 
    
    for row in (range(rows_num-1)):
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

t_list = [5,8,12]
v_list = [106.8,177.2,279.2]
A = defineA(t_list,v_list)
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
Z = forward_substitution(L,A[:,-1])
print(Z)
print("-"*60)
X = backward_substitution(U,Z)
print(X)