# Python for Data Science Very Basics

    > Math Operations with NumPy
    > Data Manipulation with Pandas
    > Visualization with Matplotlib

## Statistics
- **mean:** the average of the values.
- **median:** the middle value.
- **standard deviation:** the measure of spread, the square root of **variance**.
- **variance:** average of the squared differences from the mean.
- One standard deviation from the mean - is the values `from (mean-std) to (mean+std)`

## Math Operations with NumPy
```python 
x = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

# Size, dimentionality, shape of array
print(x[1][2]) # 6
print(x.ndim) # 2
print(x.size) # 9
print(x.shape) # (3, 3)

x = np.array([2, 1, 3])
x = np.append(x, 4) # [2, 1, 3, 4]
x = np.delete(x, 0) # Takes index
x = np.sort(x)

# Similar to python range()
x = np.arange(2, 10, 3) # [2, 5, 8]

# Reshaping the array
x = np.reshape(3, 1) # [[2], [5], [8]]

# Indexing and slicing 
# Same as python lists [-1], [0:4]

# Conditions
y = x[x<4] # Select element that are less than 4
y = x[(x>5) & (x%2==0)] # & (and), | (or)

# Operations
y = x.sum()
y = x.min() 
y = x.max()
y = x*2 # Broadcasting used

# Statistics
np.mean(x)
np.median(x)
np.var(x)
np.std(x)
```

## Data Manipulation with Pandas
- Built on top of **NumPy** = "numerical python", **Pandas** = "panel data"
- Used to read and extract data from files, transform and analyze it, calculate statistics and correlations.
- **Series** and **DataFrame**. A **Series** is essentially a column, and a **DataFrame** is a multi-dimensional table made up of a collection of Series.
```python
# Dictionary used to create DataFrame (DF)
data = {
   'ages': [14, 18, 24, 42],
   'heights': [165, 180, 176, 184]
} 

df = pd.DataFrame(data, index=['James', 'Bob', 'Amy', 'Dave']) # You can specify `index` if you want

# How to access row?
y = df.loc["Bob"] # df.loc[1]

# Indexing
z = df["ages"] # Series
z = df[["ages", "heights"]] # DataFrame, pay attention to brackets

# Slicing
# iloc[], same as in python lists
print(df.iloc[2]) # third row
print(df.iloc[:3]) # first 3 rows
print(df.iloc[1:3]) # rows 2 to 3 
print(df.iloc[-3:]) # accessing last three rows

# Conditons
z = df[(df['ages']>18) & (df['heights']>180)]
```
```python
# Reading data 
df = pd.read_csv("test.csv")

df.head() # First five rows
df.tail() # Last five rows

df.info()
df.describe() # Statistics: mean, min, max, percentiles. We can get for a single column too df['cases'].describe()

df.set_index("date", inplace=True) # Set as the index the "data" column
# inplace=True used to change the currect dataframe without assigning to new
```
```python
# Creating a column
df['area'] = df['height'] * df['width']
df['month'] = pd.to_datetime(df['date'], format="%d.%m.%y").dt.month_name()

# Droping a column
df.drop("state", axis=1, inplace=True)
# axis=1 specifies that we want to drop a column.
# axis=0 will drop a row.
```
```python
# Grouping
z = df['month'].value_counts()

z = df.groupby('month')['cases'].sum()

z = df['cases'].sum() # max(), min(), mean()
```