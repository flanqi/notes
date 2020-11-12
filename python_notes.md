## Data Manipulation

### Summary Statistics
1. Data types
```python
df.dtypes # get dtypes for all variables
```

### Data Transformation
1. Columns names
```python
cols = df.columns.tolist()[1:9] # get column names for column 1 through 8
df.rename(columns = {'original_name':'new_name'}) # rename columns
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
## ML Modeling
