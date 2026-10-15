#9:33 pm - 9:40 pm 
import numpy as np 

def power(A,x0):
    eigen_estimates = []
    x = x0.copy()
    while(1):
        y = A @ x
        eigen_estimated = np.max(y)
        eigen_estimates.append(eigen_estimated)
        x = y/eigen_estimated 
        if(len(eigen_estimates)>=2):
            if(abs(eigen_estimates[-1]-eigen_estimates[-2])<1e-6):
                break 
    return x,eigen_estimates 


def findLU(A):
    U = A.copy()
    #U = U[:,:-1]
    #print(A.shape)
    #print(np.zeros(A.shape))
    L = np.array(np.zeros(A.shape))
    #L = L[:,:-1]
    for r in range(A.shape[1]):
        L[r][r]=1
    #print(L)
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


def forword_substitution(L, C):
    solution_len = len(C)
    sol = np.array(np.zeros(len(C)))
    
    for i in range(solution_len):
        sol[i] = C[i]
        

def findInv(row_num,L,U):
    sol = np.array(np.zeros((row_num,row_num)))
    for i in range(row_num):
        iden_col = np.zeros((row_num))
        iden_col[i]=1
        Z = forward_substitution(L,iden_col)
        col = backward_substitution(U,Z)
        sol[i] = col 
    return sol.T 




# A = np.array([[8,2,0,0],[2,8,0,0],[0,0,3,1],[0,0,1,3]],dtype='float64')
# x = np.array([1,1,1,1],dtype='float64')
A = np.array([[2,1],[1,2]],dtype='float64')
x = np.array([[1],[0]],dtype='float64')

L,U = findLU(A)
A_inv = findInv(A.shape[0],L,U)
print(f"inverse A: {A_inv}")


x,eigen_estimates = power(A_inv,x)
print(f"smallest eigen vector: {x}")
print(f"Iterations:{ len(eigen_estimates)}")
print(f"Smallest eigenval:{eigen_estimates[-1]}")

x_val = np.arange(1,len(eigen_estimates)+1)
y = np.array(eigen_estimates)

import matplotlib.pyplot as plt 
plt.plot(x_val,y)
plt.grid(True)
plt.show()





