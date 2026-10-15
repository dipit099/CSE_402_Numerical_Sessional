import numpy as np

def gauss_jordan(A, b):
    n = len(b)
    Aug = np.hstack([A.astype(float), b.reshape(-1, 1).astype(float)])

    for k in range(n):
        Aug[k] = Aug[k] / Aug[k, k]          # normalize pivot row to 1
        for i in range(n):
            if i != k:
                Aug[i] -= Aug[i, k] * Aug[k]  # eliminate from every other row

    return Aug[:, -1]

A = np.array([[3, -0.1, -0.2], [0.1, 7, -0.3], [0.3, -0.2, 10]])
b = np.array([7.85, -19.3, 71.4])
print(gauss_jordan(A, b))   # -> [3. -2.5  7.]




def gauss_jordan(A, b):
    n = len(b)
    Aug = np.hstack([A.astype(float), b.reshape(-1,1).astype(float)])
    
    for k in range(n):
        Aug[k] = Aug[k] / Aug[k,k] # making the corner 1
        
        # checking if its not the same row then we are reducing the 1
        for i in range(n):
            if i != k:
                Aug[i] = Aug[i] - Aug[i, k] * Aug[k]
                
    return Aug[:, -1]





def gauss_jordan(A, b):
    n = len(b)
    Aug = np.hstack([A.astype(float), b.reshape(-1, 1).astype(float)])
    
    
    for k in range(n):
        Aug[k] = Aug[k] / Aug[k,k]
        
        for i in range(n):
            if i != k:
                Aug[i] = Aug[i] - Aug[i,k] * Aug[k]
                
            


A = np.array([3, -0.1, -0.2], [0.1, 7, -0.3], [0.3, -0.2, 10])

b = np.array([7.85, -19.3, 71.4])

print(gauss_jordan(A, b))


