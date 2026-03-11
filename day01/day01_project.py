import numpy as np

a = np.array([[1,2,3],
              [4,5,6]])
b = np.array([[7,8],
              [9,10],
              [11,12]])

def matrix_multiply(a,b):
    rows_a = a.shape[0]
    cols_a = a.shape[1]
    cols_b = b.shape[1]

    result = np.zeros((rows_a, cols_b))
    for i in range(rows_a):
        for j in range(cols_b):
            for k in range(cols_a):
                result[i,j] += a[i,k] * b[k,j]
    return result

my_result = matrix_multiply(a,b)
np_result = np.dot(a,b)

print(f'My result:\n {my_result}\n')
print(f'np_result:\n {np_result}\n')
print(f'Are both same: {np.allclose(my_result,np_result)}')