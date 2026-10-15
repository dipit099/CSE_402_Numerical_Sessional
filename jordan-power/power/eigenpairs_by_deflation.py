#9:51 pm - 10:32 pm 

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


# for i in range(row_num):
#     iden_col = np.zeros((row_num))
#     iden_col[i]=1
#     Z = forword_substituition(L, iden_col)
#     col = backward_substitution(U,Z)
#     sol[i] = col
    
# return sol.T



def findInv(row_num, L, U):
    sol = np.array(np.zeros((row_num, row_num)))
    for i in range(row_num):
        



def power(A,x0):
    #9:16 pm - 9:32 pm 
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



def power(A, x0):
    eigen_estimates = []
    
    x = x0.copy()
    
    while(1):
        y = A @ x
        eigen_estimated = np.max(y)
        eigen_estimates.append(eigen_estimated)
        
        x = y/eigen_estimated
        
        if(len(eigen_estimates) >= 2) :
            if(abs(eigen_estimates[-1] - eigen_estimates[-2] < 1e-6)):
                break
    return x, eigen_estimates




def find_all_eigens(A):
    # A = np.array([[2,1],[1,2]],dtype='float64')
    final_eigens = []
    V = []
    x = np.array(np.arange(0,len(A[0])).reshape(-1,1),dtype='float64')
    x,eigen_estimates = power(A,x)
    final_eigens.append(eigen_estimates[-1])
    V.append(x)
    print(f"largest eigen value: {eigen_estimates[-1]}, largest eigen vector: {x}")
    #print(f"x: {x}, value abs: {np.sqrt(x.T@x)}")
    for i in range(len(A[0])-1):
        v_normalized = x / (np.sqrt(x.T@x))
        #print(v_normalized@v_normalized.T)
        A = A - eigen_estimates[-1]*(v_normalized@v_normalized.T)
        x,eigen_estimates = power(A,x)
        final_eigens.append(eigen_estimates[-1])
        V.append(x)
        print(f"{i+2}th eigen value: {eigen_estimates[-1]}, {i+2}th eigen vector: {x}")
    lamda = np.array(np.zeros((len(A[0]),len(A[0]))))
    for i in range(len(A[0])):
        lamda[i][i] = final_eigens[i]
    return lamda,np.hstack(V)
     
A = np.array([[8,2,0,0],[2,8,0,0],[0,0,3,1],[0,0,1,3]],dtype='float64')
# x = np.array([1,1,1,1],dtype='float64')

#A = np.array([[2,1],[1,2]],dtype='float64')
lamda,V = find_all_eigens(A)
#print(V)
L,U = findLU(V)
print(V@lamda@findInv(A.shape[0],L,U))

# print(f"largest eigen vector: {x}")
# print(f"largest eigen value:{eigen_estimates[-1]}")
# print(f"Iterations:{ len(eigen_estimates)}")

# x_val = np.arange(1,len(eigen_estimates)+1)
# y = np.array(eigen_estimates)

# import matplotlib.pyplot as plt 
# plt.plot(x_val,y)
# plt.grid(True)
# plt.show()