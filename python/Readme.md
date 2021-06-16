# [Python for Data Science Very Basics](https://www.sololearn.com/learning/1161)

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
# We can use Python Lists to create NumPy arrays
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
```python
# https://www.sololearn.com/learning/eom-project/1161/1156
# One standart devisation from the mean
import numpy as np

data = np.array([150000, 125000, 320000, 540000, 200000, 120000, 160000, 230000, 280000, 290000, 300000, 500000, 420000, 100000, 150000, 280000])

mean_h = np.mean(data)
std_h = np.std(data)
 
low, high = mean_h - std_h, mean_h + std_h 

count = len([v for v in data if low < v < high]) 
res = count * 100 / len(data)
print(res)
```

## Data Manipulation with Pandas
- Built on top of **NumPy** = "numerical python", **Pandas** = "panel data"
- Used to read and extract data from files, transform and analyze it, calculate statistics and correlations.
- **Series** and **DataFrame**. A **Series** is essentially a column, and a **DataFrame** is a multi-dimensional table made up of a collection of Series.
- `loc` explicit indexing (own indexing), `iloc` implicit indexing (0, 1, 2, 3)
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
```python
"""COVID Data Analysis"""
import pandas as pd

df = pd.read_csv("https://www.sololearn.com/uploads/ca-covid.csv")

df.drop('state', axis=1, inplace=True)
df.set_index('date', inplace=True)

df['ratio'] = df['deaths'] / df['cases']

largest = df.loc[df['ratio'] == df['ratio'].max()] # df.loc[df['ratio'].max()] we cannot do that
print(largest)
```

## Visualization with Matplotlib
- https://www.w3schools.com/python/matplotlib_intro.asp
- **Matplotlib** is a library used to create graphs, charts, and figures. It also provides functions to customize your figures by changing the colors, labels, etc.
- **Matplotlib** works really well with **Pandas**! **Pandas** works well with **NumPy**.
```py
import matplotlib.pyplot as plt
import pandas as pd

s = pd.Series([18, 42, 9, 32, 81, 64, 3])
s.plot(kind='bar')
plt.savefig('plot.png')
```
- Data = Y axis, index = X axis. 
```py
"""Line Plot"""
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("https://www.sololearn.com/uploads/ca-covid.csv")
df.rdop('state', axis=1, inplace=True)
df['date'] = pd.to_datetime(df['date'], format="%d.%m.%y")
df['month'] = df['date'].dt.month
df.set_index('date', inplace=True)

df[df['month']==12]['cases'].plot()
# Multiple lines
# (df[df['month']==12])[['cases', 'deaths']].plot()
```
```py
"""Bar Plot"""
(df.groupby('month')['cases'].sum()).plot(kind="bar") # barh = horizontal bar
# OR
# df = df.groupby('month')
# df['cases'].sum().plot(kind="bar")
```
```py
"""Box Plot"""
df[df["month"]==6]["cases"].plot(kind="box")
```
```py
"""Histogram"""
df[df["month"]==6]["cases"].plot(kind="hist")
```
- A **histogram** is a graph showing *frequency* distributions. Similar to box plots, **histograms** show the distribution of data.
Visually histograms are similar to bar charts, however, histograms display frequencies for a group of data rather than an individual data point; therefore, no spaces are present between the bars. 
```py
"""Area Plot"""
df[df["month"]==6][["cases", "deaths"]].plot(kind="area", stacked=False)
```
```py
"""Scatter Plot"""
df[df["month"]==6][["cases", "deaths"]].plot(kind="scatter", x='cases', y='deaths')
```
```py
"""Pie Chart"""
df.groupby('month')['cases'].sum().plot(kind="pie")
```
```py
"""Plot formatting"""
df[['cases', 'deaths']].plot(kind="area", legend=True, stacked=False, color=['#1970E7', '#E73E19'])
plt.xlabel('Days in June')
plt.ylabel('Number')
plt.suptitle("COVID-19 in June")
```

## Sololearn Certificate
<img src="https://www.sololearn.com/certificates/course/en/13739122/1161/landscape/png">
