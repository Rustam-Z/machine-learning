# Python Data Science Handbook

Rustam-Z🚀 • 1 June 2021

My notes on **NumPy**, **Pandas**, **Matplotlib**, and **Scikit-Learn** 

## Contents
1. IPython: Beyond Normal Python
2. Introduction to NumPy: Math operations with Python
    - Creating Arrays
    - The Basics of NumPy Arrays
    - Computation on NumPy Arrays
3. Data Manipulation with Pandas
4. Visualization with Matplotlib
5. Machine Learning

## Introduction to NumPy

### Creating Arrays
```python
np.zeros(10, dtype=int) # Create a length-10 integer array filled with zeros
np.ones((3, 5), dtype=float) # Create a 3x5 floating-point array filled with 1s
np.full((3, 5), 3.14) # Create a 3x5 array filled with 3.14
np.arange(0, 20, 2) # As python's range()
np.linspace(0, 1, 5) # Create an array of five values evenly spaced between 0 and 1
np.random.random((3, 3)) # 3x3 array, random values between 0 and 1
np.random.normal(0, 1, (3, 3)) # normal distribution, with mean 0 and standard deviation 1
np.random.randint(0, 10, (3, 3)) # random integers between 0 and 10
np.eye(3) # Create a 3x3 identity matrix
np.empty(3) # Create an uninitialized array of three integers

np.zeros(10, dtype='int16') # same as 
np.zeros(10, dtype=np.int16)
```

### The Basics of NumPy Arrays
- *Attributes of arrays*
    - Determining the size, shape, memory consumption, and data types of arrays
- *Indexing of arrays*
    - Getting and setting the value of individual array elements
- *Slicing of arrays*
    - Getting and setting smaller subarrays within a larger array
- *Reshaping of arrays*
    - Changing the shape of a given array
- *Array Concatenation and Splitting*
    - Combining multiple arrays into one, and splitting one array into many

```python
""" Attributes of arrays """
x = np.random.randint(10, size=(3, 4, 5)) # Three-dimensional array
x.ndim # 3
x.shape # (3, 4, 5)
x.size # 60 = 3*4*5
x.dtype # dtype: int64
x.nbytes # total size of array in bytes
```
```python
""" Indexing of arrays """
# Same as in python lists, but beware if you insert float into int, the result will be int
x[0][0][1] or x[0, 0, 1]
```
```python
""" Slicing of arrays """
# Same as python lists
# NOTE! Multidimensional slices work in the same way, with multiple slices separated by commas.
x[start:stop:step]
```
- NOTE! NumPy arrays return the *view* of original array after slicing. So, when we modify our sliced array it will affect to original array. Use **copy()** method when you don't want it. `x_copy = x[:2, :2].copy()`

```python
""" Reshaping of Arrays """
# reshape() method
np.arange(1, 10).reshape((3, 3))
```
```python
""" Array Concatenation """
x = np.array([1, 2, 3])
y = np.array([3, 2, 1])

grid = np.array([[9, 8, 7],[6, 5, 4]])

np.concatenate([x, y]) # axis=1 same as x axis, then it will concatenated horizontally

# If working with different dimensions
np.vstack([x, grid])
np.hstack([grid, y])
# np.dstack will stack arrays along the third axis

""" Splitting of arrays """
# np.split, np.hsplit, np.vsplit
x = [1, 2, 3, 99, 99, 3, 2, 1]
x1, x2, x3 = np.split(x, [3, 5]) # we give splitting points
print(x1, x2, x3) # [1 2 3] [99 99] [3 2 1] # N --> N+1 subarray
```

### Computation on NumPy Arrays
- *unary ufuncs*, operate on a single input, and *binary ufuncs*, operate on two inputs
```
+       np.add          Addition (e.g., 1 + 1 = 2) 
-       np.subtract     Subtraction (e.g., 3 - 2 = 1) 
-       np.negative     Unary negation (e.g., -2) 
*       np.multiply     Multiplication (e.g., 2 * 3 = 6) 
/       np.divide       Division (e.g., 3 / 2 = 1.5)
//      np.floor_divide Floor division (e.g., 3 // 2 = 1)
**      np.power        Exponentiation (e.g., 2 ** 3 = 8) 
%       np.mod          Modulus/remainder (e.g., 9 % 4 = 1)

        np.abs(x)
        np.sin(x), np.cos(x), np.tan(x)
        np.log(x), np.log2(x), np.log10(x)
        np.exp(x)        e^x
        np.exp2(x)       2^x
        np.power(3, x)   3^x
        np.expm1(x)      exp(x) - 1
        np.log1p(x)      log(1 + x)

        
```