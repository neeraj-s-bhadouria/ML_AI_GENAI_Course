import numpy as np

# 1-D array
data = np.array([1,2,3,4,5])

print(f'Array: {data}')
print(f'Type: {type(data)}')
print(f'Shape: {data.shape}')
print(f'DataType: {data.dtype}\n')


# 2-D Array - Matrix
data = np.array([[1,2,3],
                 [4,5,6],
                 [7,8,9]
])

print(f'Matrix: {data}')
print(f'Shape: {data.shape}')
print(f'Total elements: {data.size}')
print(f'Dimensions: {data.ndim}\n')


# creating different arrays from inbuilt methods
zeroes = np.zeros((3,3))                            # creating a 3x3 array with zeroes
ones = np.ones((2,4))                               # creating a 2x4 array with 1
random = np.random.rand(3,3)                        # creating a 3x3 array with random values between 0 and 1
sequence = np.arange(0,10,2)  # creating a 1-D array from elements 0 to 10 with every 2nd number

print(f'Zeroes\n: {zeroes}\n')
print(f'Ones\n: {ones}\n')
print(f'Random\n: {random}\n')
print(f'sequence\n: {sequence}\n')


# most important methods of numpy for daily use in ML

a = np.array([1,2,3,4,5])
b = np.array([10,20,30,40,50])
# Basic ops
print(f'Addition: {a+b}')
print(f'Multiplication: {a*b}')
print(f'Square: {a**2}')
print(f'Square root: {np.sqrt(a)}\n')
# Mostly used in ML
print(f'Mean: {np.mean(a)}')
print(f'Sum: {np.sum(a)}')
print(f'Max: {np.max(a)}')
print(f'Min: {np.min(a)}')


# Matrix Multiplication
a = np.array([[1,2],
              [3,4]])
b = np.array([[5,6],
              [7,8]])

print(f'Element-wise Multiplication:\n {a*b}\n')
print(f'Matrix Multiplication:\n {np.dot(a,b)}\n')
print(f'@ operator:\n {a@b}\n')                             # modern way - same as dot


# indexing and slicing
arr = np.array([[1,2,3],
                [4,5,6],
                [7,8,9]])

print(f'Row 0, col 1: {arr[0,1]}')                          # Indexing a single element
print(f'Row 1: {arr[1,]}')                                  # slicing the whole first row
print(f'Col 2: {arr[:,2]}')                                  # slicing the whole 2nd column
print(f'Top-left 2x2:\n {arr[0:2, 0:2]}')                   # Subset - top-left 2x2