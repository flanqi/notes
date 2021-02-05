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

### Indexing

```python
df['col'] = df.col # returns a pd.series
df[['col']] # returns a pd.dataframe
```
NOTE: By using df.col, the column name cannot be a reserved word for python, and it cannot start with numbers, etc. This method of selecting data is called attribute access.

```python
l = [1, 2, 3]
l[-1] # last element 3
```

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

0. Set Up
set up parameters:
```python
# dpi
plt.rcParams['figure.dpi'] = 150

# figure size
fig = plt.figure(figsize=(4,3))
```

legend location:
```python
# matplotlib
plt.legend(loc='lower right')

# sns
g = sns.scatterplot(...)
_ = g.legend(loc='center right')
```

subplots - figure and axes:
```python
fig, ax = plt.subplots(2,1) # 2 rows, 1 column; fig refers to the canvas, ax is used to add contents and set labels etc.
ax1 = ax[0,0]
ax2 = ax[1,0]
# ax1.scatter(x,y) #add scatterplot to first plot

# equivalently
fig = plt.figure(figsize=(4,3))
ax1 = plt.subplot(2,1,1) # 2 rows, 1 column, the first plot
ax2 = plt.subplot(2,1,2) # the second plot
```
see more about this [here](https://stackoverflow.com/questions/34162443/why-do-many-examples-use-fig-ax-plt-subplots-in-matplotlib-pyplot-python).

set plot labels & titles:
```python
ax.set_xlabel('xlabel')
ax.set_ylabel('ylabel')

ax.set_title(title')
```

1. Line charts color by group
```python
fig, ax = plt.subplots(figsize=(8,5))

for idx, gp in df.groupby('group_variable'):
    gp.plot(x='x_var', y='y_var', ax=ax, label=idx)

plt.show()
```
2. Reference Lines
```python
_ = g.axvline(x)
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

5. Scatterplots

```python
g = sns.scatterplot(x='xvar', y='yvar', hue='category', data=df)
```

paired scatterplots:
```python
sns.pairplot(
    data=df,
    vars=['var1','var2','var3','var4']
)
```
## ML Modeling
### Linear Regression
prepare data:
```python
import statsmodel.api as sm

# prepare Y and X
y = df['target']
X = y.drop('target', axis=1)
X = (X-X.mean())/X.std()
X = sm.add_constant(df[['var1','var2']], prepend=True) # add constant for intercept
```

modeling
```python
lm = sm.OLS(y, X).fit()
lm.summary()
```

vifs:
```python
from statsmodels.stats.outliers_influence import variance_inflation_factor

vifs = pd.Series({
    X.columns[i]:variance_inflation_factor(X.values,i) for i in range(1,X.shape[1])
})
vifs
```

fitted plot:
```python
fig, ax = plt.subplots(1,1,figsize=(4,3))
_ = sns.scatterplot(x=lm.fittedvalues, y=lm.resid,ax=ax)
_ = ax.set_xlabel('Fitted')
_ = ax.set_ylabel('Residuals')
```

qq plot:
```python
from scipy.stats import probplot

fig, ax = plt.subplots(1,1,figsize=(4,3))
_ =  probplot(lm.resid_pearson, plot=ax)
_ = ax.set_ylabel('Standardized residuals')
_ = ax.set_title('Normal Q-Q plot')
```

### Logisic Regression
prepare data:
```python
import statsmodel.api as sm

# prepare Y and X
y = df['target']
X = sm.add_constant(df[['var1','var2']], prepend=True) # add constant for intercept
```

modeling:
```python
glm_binom = sm.GLM(y, X, family = sm.families.binomial()).fit()
print(glm_binom.summary())
```

evaluation:
```python
glm_binom.params # coeff
cov = glm_binom.cov_params() # covariance matrix for coeff
variances = np.diag(cov) 
std_err = pd.Series(np.sqrt(variances), index = cov.columns) # standard errors for coeff

glm_binom.tvalues # z-scores/t-scores

print('AIC: %.3f' % glm_binom.aic)
print('log-likelihood: %.3f' % -glm_binom.deviance/2
```

predication:
```python
p_hat = glm_binom.predict(newdata) # pred prob
df['class'] = 1*(glm_binom.predict(X) > cut_off)

# calculate log-likelihood manually
log_lik = np.sum(y*np.log(p_hat)+(1-y)*np.log(1-p_hat))
```
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

model = KMeans(n_clusters=5, tol=1e-6, n_init=100, random_state=42)
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
