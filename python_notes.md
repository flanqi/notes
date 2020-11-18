## Table of contents

* [Set Up](#set-up)
* [Data Manipulation](#data-manipulation)
* [Data Visualization](#data-visualization)
* [ML Modeling](#ml-modeling)


## Set Up
1. Filter out warnings
```python
import warnings
warnings.filterwarnings('ignore')
```
2. Common packages for DS
```python
import pandas as pd 
import numpy as np
# visualizations
import matplotlib.pyplot as plt
%matplotlib inline # make your plot appear and stored in jupyter notebook
import seaborn as sns
```
## Data Manipulation

### Subset Dataframe

```python
df['col'] = df.col # returns a pd.series
df[['col']] # returns a pd.dataframe
```
NOTE: By using df.col, the column name cannot be a reserved word for python, and it cannot start with numbers, etc. This method of selecting data is called attribute access.

Read more about [Indexing and Selecting with pandas](https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html).

### Summary Statistics
1. Data types
```python
df.dtypes # get dtypes for all variables
```

### Data Transformation
1. Columns 
```python
cols = df.columns.tolist()[1:9] # get column names for column 1 through 8
df.rename(columns = {'original_name':'new_name'}) # rename columns
df.drop(columns=['col1','col2']) # drop columns by name
```
2. Add proportion by group to df 
```python
def prop(group):
    group['prop']=group.Count/group.Count.sum()
    return group
    
df.groupby('group_variable').apply(prop)
```
### Substrings 
1. Substring for string 
```python
x = "hello"
x[1:] # output: "ello"
```
2. Substring a column
```python
df['col'] = df['col'].str[:-3] # "12:00EST" -> "12:00"
```
### Date & Time
```python
df['col'] = pd.to_datetime(df['col'], format='%Y-%m-%d %H:%M:%S')
```

### Missing Values
1. replace 0 with nas (or vice versa)
```python
df.replace(0, np.nan, inplace=True)
```
2. Drop nas
```python
df.dropna(inplace=False)
```
## Data Visualization
1. Line charts color by group
```python
fig, ax = plt.subplots(figsize=(8,5))

for idx, gp in df.groupby('group_variable'):
    gp.plot(x='x_var', y='y_var', ax=ax, label=idx)

plt.show()
```
2. Legend position
```python
plt.legend(loc='lower right')
```
3. Heatmap for correlations
```python
corr = df[cols].corr()

# Generate a mask for the upper triangle
mask = np.triu(np.ones_like(corr, dtype=bool))

# Set up the matplotlib figure
f, ax = plt.subplots(figsize=(11, 9))

# Generate a custom diverging colormap
cmap = sns.diverging_palette(230, 20, as_cmap=True)

# Draw the heatmap with the mask and correct aspect ratio
sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0,
            square=True, linewidths=.5, cbar_kws={"shrink": .5})
```
4. Heatmap for NAs
```python
colours = ['seashell', 'coral'] # customizable
f, ax = plt.subplots(figsize = (12,8))
sns.set_style("whitegrid")
plt.title('Missing Values Heatmap', )
sns.heatmap(df2[cols].isna(), cmap=sns.color_palette(colours));
```
## ML Modeling
