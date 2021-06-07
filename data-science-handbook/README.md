# Python Data Science Handbook

Rustam-Z🚀 • 1 June 2021

My notes on **NumPy: ndarray**, **Pandas: DataFrame**, **Matplotlib**, and **Scikit-Learn** 

## Contents
1. IPython: Beyond Normal Python - *All features of Jupyter Notebook*
2. Introduction to NumPy: Math operations with NumPy
    - Creating Arrays
    - The Basics of NumPy Arrays
    - Computation on NumPy Arrays
    - Fancy indexing
    - Structured Arrays
3. Data Manipulation with Pandas
4. Visualization with Matplotlib
5. Machine Learning

## CHAPTER 2: Introduction to NumPy
- `axis=0 is column`, `axis=1 is row`

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

- indices `(e.g., arr[0])`, slices `(e.g., arr[:5])`, and boolean masks `(e.g., arr[arr > 0])`
- [np.newaxis()](https://stackoverflow.com/questions/46334014/np-reshapex-1-1-vs-x-np-newaxis)

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
```python
x = np.arange(1, 6)
np.add.reduce(x) # 15, sum of all elements
np.multiply.reduce(x) # 120, mulitplication of all elements

np.add.accumulate(x) # array([ 1, 3, 6, 10, 15]), intermediate result
np.multiply.accumulate(x) # array([ 1, 2, 6, 24, 120])

np.multiply.outer(x, x) # N+1 dimension multiplication

np.sum          Compute sum of elements
np.prod         Compute product of elements
np.mean         Compute median of elements
np.std          Compute standard deviation
np.var          Compute variance
np.min          Find minimum value
np.max          Find maximum value
np.argmin       Find index of minimum value
np.argmax       Find index of maximum value
np.median       Compute median of elements
np.percentile   Compute rank-based statistics of elements   np.percentile(arr, 25))
np.any          Evaluate whether any elements are true
np.all          Evaluate whether all elements are true
```
```python
"""Comparison Operators"""
==      np.equal
!=      np.not_equal
<       np.less             np.less(x, 3) is x < 3
<=      np.less_equal
>       np.greater
>=      np.greater_equal

# Example
x = np.array([1, 2, 3, 4, 5])
x < 3  # array([ True, True, False, False, False], dtype=bool)
(2 * x) == (x ** 2)  # array([False, True, False, False, False], dtype=bool)
```
```python
"""Working with Boolean Arrays"""
print(x)  # [[5 0 3 3][7 9 3 5][2 4 7 6]]

# Counting entries
np.count_nonzero(x < 6) # 8, how many values less than 6?
np.sum(x < 6) # 8, counts elements less than 6
np.sum(x < 6, axis=1) # how many values less than 6 in each row?
np.any(x > 8) # are there any values greater than 8?
np.all(x < 10) # are all values less than 10?
np.all(x < 8, axis=1) # are all values in each row less than 8?

# Boolean operators
&   np.bitwise_and
|   np.bitwise_or
^   np.bitwise_xor
~   np.bitwise_not
np.sum((inches > 0.5) & (inches < 1)) # that's counts the number of elements
np.sum(~( (inches <= 0.5) | (inches >= 1) ))

x[x < 5] # [0 3 3 3 2 4]

# Fancy indexing 
x = rand.randint(100, size=10)
y = np.array([1, 2])
x[y] # array([92, 14])
```
- `np.sort(x)`, `np.argsort(x)` , `np.sort(X, axis=0)` = sort each column of X
- Partial Sorts: `np.partition(x, 3)` - returns 2 smallest elements to the left

```python
"""NumPy’s Structured Arrays: Compound data types"""
name = ['Alice', 'Bob', 'Cathy', 'Doug']
age = [25, 45, 37, 19]
weight = [55.0, 85.5, 68.0, 61.5]

# We need to combine them
x = np.zeros(4, dtype=int)
data = np.zeros(4, dtype={'names':('name', 'age', 'weight'), 'formats':('U10', 'i4', 'f8')})
data['name'] = name
data['age'] = age
data['weight'] = weight

print(data) # [('Alice', 25, 55.0) ('Bob', 45, 85.5) ('Cathy', 37, 68.0) ('Doug', 19, 61.5)]

# Get all names
data['name'] # array(['Alice', 'Bob', 'Cathy', 'Doug'], dtype='<U10')

# Get first row of data
data[0] # ('Alice', 25, 55.0)

# Get the name from the last row
data[-1]['name'] # 'Doug'

# Get names where age is under 30
data[data['age'] < 30]['name'] #  array(['Alice', 'Doug'], dtype='<U10')

"""Creating Structured Arrays"""
tp = np.dtype({'names':('name', 'age', 'weight'), 'formats':('U10', 'i4', 'f8')})
tp =  np.dtype([('name', 'S10'), ('age', 'i4'), ('weight', 'f8')])
tp =  np.dtype('S10,i4,f8')

# Then assign in dtype argument:
X = np.zeros(1, dtype=tp)

"""More advanced compound arrays"""
tp = np.dtype([('id', 'i8'), ('mat', 'f8', (3, 3))])
X = np.zeros(1, dtype=tp)
print(X[0])
print(X['mat'][0])
```

## CHAPTER 3: Data Manipulation with Pandas
- Consists of **Series** and **DataFrame** objects, also **Index**
- `TIP!` “Explicit is better than implicit"

```python
"""The Pandas Series Object"""
# pd.Series(data, index=index)
data = pd.Series([0.25, 0.5, 0.75, 1.0]) # Series = 1D array
data.index
data.values
# Difference: Numpy = implicitly defined index, but Pandas Series = explicitly defined index (obvious, can be changed)
data = pd.Series([0.25, 0.5, 0.75, 1.0], index=[2, 'b', 'c', 'd'])
```
```python
"""The Pandas DataFrame Object"""
# It consists of Series
states = pd.DataFrame({'population': population, 'area': area})
# pd.DataFrame([{'a': 1, 'b': 2}, {'b': 3, 'c': 4}]) # From a list of dicts
# A = np.zeros(3, dtype=[('A', 'i8'), ('B', 'f8')]) # From a NumPy structured array
```
- Pages 102-105 **DataFrame** object creating variations
```python
"""Index object"""
# The index object is immutablle so that it cannot be changed after diclaration
ind = pd.Index([2, 3, 5, 7, 11])
ind[1] = 0 # ERROR

indA = pd.Index([1, 3, 5, 7, 9])
indB = pd.Index([2, 3, 5, 7, 11])
indA & indB # intersection => Int64Index([3, 5, 7], dtype='int64')
indA | indB # union => Int64Index([1, 2, 3, 5, 7, 9, 11], dtype='int64')
indA ^ indB # symmetric difference => Int64Index([1, 2, 9, 11], dtype='int64')
```
```python
"""Data Selection in Series"""
data = pd.Series([0.25, 0.5, 0.75, 1.0], index=['a', 'b', 'c', 'd'])

data['b'] # 0.5
'a' in data # True
data.keys()
data.items() # key: value
data['e'] = 1.25 # We can add new item

# slicing explicit, 'c' will be included 
data['a':'c'] 

# slicing implicit
data[0:2] 

# masking
data[(data > 0.3) & (data < 0.8)] 

# fancy indexing
data[['a', 'e']]

"""Indexers: loc, iloc, and ix
loc = allows indexing and slicing that always references the explicit index (own indexing)
iloc = allows indexing and slicing that always references the implicit Python-style index (from 0)

`TIP!` “Explicit is better than implicit"
"""
```
```python
"""Data Selection in DataFrame"""
# DataFrame as a dictionary
data = pd.DataFrame({'area':area, 'pop':pop})
data['area']
data.area # if name == str method then not working
# Add new column
data['density'] = data['pop'] / data['area']
# Access samples
data.loc['Texas']

# DataFrame as two-dimensional array
data.values 
data.T # Transpose
data.iloc[:3, :2] # Chooses both row and column respectively
data.loc[:'New York', :'pop'] # same as previous
data.loc[data.density > 100, ['pop', 'density']] # fancy indexing
# Change like this
data.iloc[0, 2] = 90
data[data.density > 100]
```
- Until page 114
- We can perform NumPy operations over Pandas Series and Dataframe (adding, division)
```py
A = pd.Series([2, 4, 6], index=[0, 1, 2])
B = pd.Series([1, 3, 5], index=[1, 2, 3])
print(A + B)
print(A.add(B, fill_value=0)) # the set which doesn't include that index will be replaces with 0 

## A.add(B)
+ add()
- sub(), subtract()
* xmul(), multiply() 
/ truediv(), div(), divide()
// floordiv()
% mod()
** pow()
```
```py
"""Missing Data in Pandas"""
vals2 = np.array([1, np.nan, 3, 4])
np.nansum(vals2), np.nanmin(vals2), np.nanmax(vals2)

# NaN and None in Pandas
x = pd.Series(range(2), dtype=int)
x[0] = None # Then it will be represented as NaN in DataFrame

"""Operating on Null Values"""
isnull() # True / False for each element
notnull() # opposite of isnull()
dropna() # Return a filtered version of the data
fillna()

df.isnull()
data[data.notnull()]
data.dropna()
df.dropna(axis='columns') # df.dropna(axis=1) | how='all', by default how='any'
```