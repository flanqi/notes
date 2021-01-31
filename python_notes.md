## Table of contents

* [Set Up](#set-up)
* [Data Manipulation](#data-manipulation)
* [Data Visualization](#data-visualization)
* [ML Modeling](#ml-modeling)
    * [Clustering](#clustering)


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
Common matplotlib parameters.


| parameter     | description             |
| ------------- |:-----------------------:|
| s             | size of marker          | 
| c             | color                   |


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
### Clustering
#### K-Means Clustering
Preprocessing
```python
# standardization
from sklearn import preprocessing
df['Zvar'] = preprocessing.scale(df.var)
```

```python
from sklearn.cluster import KMeans

model = KMeans(c_clusters=5, tol=1e-6, n_init=100, random_state=42)
model.fit(df)
print(model.cluster_centers_) # means of clusters

pd.crosstab(index=model.labels_, columns="count") # counts of clusters
df['cluster'] = model.labels_ # add assignments to df
df.boxplot('var', by='cluster') # profile on 'var'
model.inertia_ # SSE, total within-cluster error

# plots
plt.scatter(df['var1'], df['var2'], c=model.labels_) # 2-d cluster plot
centers = model.cluster_centers_
plt.scatter(centers[:,0], centers[:,1], c=[0,1,2,3,4], s=200, alpha=0.5) # add centers to clusters
plt.show()
```

Profiling
```python
# anova test on var
import statsmodels.api as sm
from statsmodels.formula.api import ols

fit = ols('var ~ C(model)', data=df).fit() 
table = sm.stats.anova_lm(fit, typ=2)
print(table)

# run means of var by cluster using dplyr
from dfply import *
(df >> group_by(X.model) >> summarize(meanvar = X.var.mean()))
# or
df.groupby('cluster')['var'].mean()

# profile categorical variable
pd.crosstab(index=df['cluster'], columns=df['var'], normalize='columns')
pd.crosstab(index=df['cluster'], columns=df['var'], normalize='index')

smtab = sm.stats.Table.from_data(df[['cluster', 'var']])
print(smtab.table_orig) # counts
print(smtab.fittedvalues) # predicted values under independence
print(smtab.resid_pearson) # residuals
print(smtab.test_nominal_association().pvalue) # chi-square test p-value
```

#### Gaussian Mixture

Choose the model type (covariance_type):
1. *spherical*： all spherical with different variance/volume (VII)
2. *diag*: variable volume, variable shape, axes orientation (VVI)
3. *tied*: equal volume, equal shape, equal orientation, with rotation (EEE)
4. *full*: all parameters variable, with rotation (VVV)

```python
from sklearn import mixture

gmm = mixture.GaussianMixture(n_components=2, covariance_type='full').fit(df)

# means
gmm.means_

# variances
gmm.covariances_

# within-cluster standard deviations
np.sqrt(gmm.covariances_)

# prior probabilities
gmm.weights_
```
