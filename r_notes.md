## Table of contents
* [Data Manipulation](#data-manipulation)
* [ML Modeling](#ml-modeling)
  * [Logistic Regression](#logistic-regression)
  * [Discriminant Analysis](#discriminant-analysis)
  * [GLMs](#glms)
## Data Manipulation
### Subset & Index
See [tutorial](https://www.statmethods.net/management/subset.html).

```r
# get col names
names(df)
```

```r
df$Sepal.Length #dataframe
df['Sepal.Length'] #dataframe
df[['Sepal.Length']] #vector
```

```r
# select by col names
cols = c(paste0('v',1:3))
df[cols]
df[c('Sepal.Length','Petal.Length')]

# select by col number
df[c(1,3)]
df[c(-3,-5)] # drop 3rd, 5th col

# select by boolean
c = names(df) %in% c('Sepal.Length','Sepal.Width') # returns a list of booleans
df[c]
df[!c]

# select by index
df[1:3] # subset cols
df[1:5,]
df[,1:2]
df[1:5,1:2]
```

```r
# delete col
df = iris
df$Sepal.Length = NULL # this doesn't change original df
```

```r
# filter rows by condition
df[which(df$Sepal.Length>5 & df$Species=="virginica"),] # use | for 'or'

# or equivalently
attach(df)
newdata <- df[ which(Sepal.Length>5 & Species=="virginica"), ]
detach(df)
```

```r
# filter by subset function
subset(df, Sepal.Length>5 | Sepal.Width>5, select=c(Sepal.Length, Species))
subset(df, Sepal.Length>5 | Sepal.Width>5, select=Sepal.Length:Petal.Width)
```

```r
# select & filter by dplyr
library(dplyr)

df %>% select(Sepal.Length:Petal.Width) %>% filter(Sepal.Length > 5) # select cols and filter rows

# select by col name
df %>% select(v1, v2, v3)

# select by list
df %>% select(c(Septal.Width, Petal.Width)) # with or without quotes
df %>% select(c(1, Petal.Width))

# special functions for select
df %>% select(starts_with('Sepal'))
df %>% select(ends_with('Width'))
df %>% select(contains('Sepal'))
```

```r
# random sample
mysample <- df[sample(1:nrow(mydata), 50,
   replace=FALSE),] # without replacement
```

see more about [select](https://dplyr.tidyverse.org/reference/select.html) and [filter](https://dplyr.tidyverse.org/reference/filter.html) in dplyr.

## ML Modeling
### Logistic Regression
#### Binary Logistic Regression
```r
# fit the model
model = glm(Y ~ X, family = binomial, data = df) 

# prediction
predict = predict(model, newdata = test, type = "response") 

# confusion matrix
table(test$Y, predict > cut_off_probability)
```
ROC Curve:
```r
library(pROC)
plot.roc(df$Y, model$fitted.values, print.auc=T, xlab="Specificity", ylab="Sensitivity")
```

#### Nominal Logistic Regression
```r
# fit the model
library(nnet)
model = multinom(Y ~ X, data = df, maxit = 1000) 

# prediction
predict = predict(model, newdata = test, type = "probs") 
```

#### Ordinal Logistic Regression
```r
# fit the model
library(ordinal)
df$Y = as,ordered(df$Y) # default order
df$Y = ordered(df$Y, levels = c(A,B,C), labels = c(A,B,C)) # set new order
model = clm(Y ~ X, data = df) 

# prediction
predict = predict(model, newdata = test) 
```
### Discriminant Analysis

### GLMs

