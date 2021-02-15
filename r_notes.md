## Table of contents
* [Data Manipulation](#data-manipulation-python)
* [Data Visualization](#data-visualization-python)
* [Basic ML](#basic-ml)
  * [Basic Statistics](#basic-statistics)
  * [Linear Regression](#linear-regression-python)
  * [Logistic Regression](#logistic-regression-python)
  * [GLMs](#glms)
  * [Discriminant Analysis](#discriminant-analysis)
  * [Survival Analysis](#survival-analysis)
* [Clustering](#clustering-python)
  * [KMeans Clustering](#kmeans-clustering-python)
  * [Hierarchical Clustering](#hierarchical-clustering-python)
* [Deep Learning](#deep-learning)
## Data Manipulation [Python](./python_notes.md#data-manipulation-r)

<div align="right">
    <b><a href="#table-of-contents">↥ back to top</a></b>
</div>

### Read data
```r
df = read.csv("data.csv", skip = 1, stringAsFactors = F) # skip first row of data
```
### Data Types
```
class(object) # returns the datatype of the object
df$col = parse_number(df$col) # parse the column type to numberm, needs tidyverse library
```
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
### Missing Values
See the imputation.r in r_programs folder for imputation techniques.

```r
df[complete.cases(df), ] # drop all rows with nas
```
### Substrings
```r
substr("hello",1,4) # -> "hell"
substr(df$Species,1,4) 
```
## Data Visualization [Python](./python_notes.md#data-visualization-r)

<div align="right">
    <b><a href="#table-of-contents">↥ back to top</a></b>
</div>

Click [here](https://www.statmethods.net/advgraphs/parameters.html) to see common r plot parameters.

*Examples*.

| parameter     | description             |
| ------------- |:-----------------------:|
| cex           | size of text and simbol | 
| pch           | shape of points         |
| lty           | line type               |
| lwd           | line width              |
| col           | color                   |

### Scatterplot

```r
plot(data$x, data$y,
     xlab="x variable name", ylab="y variable name",
     main="title"
     )
     
# ggplot
ggplot(df, aes(x=wt, y=mpg)) +
  geom_point(size=2, shape=23)
```

Matrix Scatterplot of Multiple Variables
```r
pairs(df, cex = 0.5, pch = 16) 
```

## Basic ML [Python](./python_notes.md#basic-ml-r)

<div align="right">
    <b><a href="#table-of-contents">↥ back to top</a></b>
</div>

### Basic Statistics
```r
sd(list) # standard deviation
cor(df) # correlation matrix
cor.test(df$v1, df$v2) # correlation test
```
### Linear Regression [Python](./python_notes.md#linear-regression-r)
```r
# fit the model
model = lm(Y ~ ., data = df) 
summary(model) # summary statistics

# prediction
predict(model, newdata, 
        interval = c("none", "confidence", "prediction"),
        level = 0.95)
```
Other statistics:

```r
# detect outliers
rstandard(model) # standardized residuals (>3)
rstudent(model)

# detect influential points
hatvalues(model) # Leverage rule >2(p+1)/n
cooks.distance(model) # cook's distance 
```

```r
library(car)
vif(model) # multicolinearity >10 for most variables
```
Plots:
```r
plot(model,which=1) # residual vs fitted
plot(model,which=2) # qq plot
plot(model,which=4) # cook's distance
```
#### Ridge/Lasso/Elastic Net Regression
```r
library(glmnet)

y = mpg$mpg
x = model.matrix(mpg~.,mpg)

# set.seed(1)
ridgecv=cv.glmnet(x, y, alpha=0,lambda=seq(0,5,0.001),nfold=3) # lasso alpha = 1
ridgecv$lambda.min # best lambda
```

```r
small.lambda.index <- which(ridgecv$lambda == ridgecv$lambda.min)
small.lambda.betas <- coef(ridgecv$glmnet.fit)[,small.lambda.index]
print(small.lambda.betas)
```

### Logistic Regression [Python](./python_notes.md#logistic-regression-r)
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

Find best cut-off probability:
```r
prob = c()
CCRs = c() # choose your own metric, here I use CCR/accuracy

for (p in seq(0,1,0.01)){
  prob = c(prob,p)
  tab=table(df$y, predict(model, df, type = "response")>p)
  CCR=sum(diag(tab))/sum(tab)
  CCRs = c(CCRs, CCR)
}

p_star <- prob[which.max(CCRs)] # the optimal p

tab = table(radiation$y, probs>p_star) # the final confusion matrix

# other metrics
sensitivity = tab[2,2]/sum(tab[2,])
specificity = tab[1,1]/sum(tab[1,])
precision = tab[2,2]/sum(tab[,2])
recall = sensitivity
f1 = 1/(0.5*((1/precision)+(1/recall)))

cbind('sensitivity'=sensitivity, 'specificity' = specificity,
      'F1' = f1)
```

#### Nominal Logistic Regression
```r
library(nnet)

# reset the reference group
df$Y = factor(df$Y)
df$Y <- relevel(df$Y, reference_level)

# fit the model
model = multinom(Y ~ X, data = df, maxit = 1000) 

# prediction probabilities
predict = predict(model, newdata = test, type = "probs") 
# predict = fitted(model, outcome=F) # predictions for training data

# prediction for groups
pred_class = rep(0,nrow(test))

for(i in 1:nrow(test)){
  pred_class[i] = which.is.max(predict[i,]) # this outputs the col number with max prob
}
pred_class
```

Confusion Matrix:
```r
library(caret)
cfm= confusionMatrix(factor(test$Y), factor(pred_class))
cfm
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
### GLMs
to be finished.

### Discriminant Analysis
#### Fisher Discriminant Function
```r
library(MASS)
iris = read.csv("Iris.csv")
fit = lda(Species_name~., data=iris[2:6], prior = c(1/3,1/3,1/3))
fit
```

Predict:
```r
predict(fit,newdata=data.frame(Petal_width=1.5,Petal_length=4,
Sepal_width=3,Sepal_length=5.5))$posterior
```

### Survival Analysis

```r
library(survival)
fit = survfit(Surv(t, churn) ~ 1. data = df, weight = count) # df is not in long-form, at each customer time t, the number of customers churned / censored are stored in variable "count"
summary(fit)
```

Survival Function Plot (this model gives the same results as in KM model):
```r
plot(fit)
```

survfit doesn't provide harzard and pdf plots, but we can extract results from the summary table.
```r
# table
res = summary(fit)
cols = lapply(c(2:6,8:11), function(x) res[x])
table = do.call(data.frame, cols)

# harzard rate plot
plot(table$time, table$n.event/table$n.risk, type = 'l', xlab = 'time', ylab = 'harzard rate')

# pdf plot
plot(table$time, (table$n.event/table$n.risk)*lag(table$surv,default = 1), type = 'l', xlab = 'time', ylab = 'pdf')

# survival function plot
plot(table$time, table$surv, type = 'l', xlab = 'time', ylab = 'survival function')
```
#### Stratified Survival Analysis

```r
# seperate survival model for different type of service length
fit = survfit(Surv(t, churn) ~ servicelen, data = df, weight = count)

# survival function plot
plot(fit, lty = 1:3) # suppose we have 3 types of service length 
legend(0.5,0.3,c('1 month', '6 months', '12 months'), lty=1:3, cex=0.8)
```
we can use the same way to extract summary table and plot harzard rates and pdf (in this case, we need to separate the tables into 3 groups).

#### Discrete Time Model

In discrete time model, we need to convert the data into long format, i.e, each (or each group of) customer has one row at each customer time t.

Turn data into long format:
```r
df_long = survSplit(data = df, cut = 0:12, end = 't', event = 'churn')
# here cut is usually 0 : max(t)
```

Simple Retention Model (r_t = r)
```r
fit = glm(churn ~ 1, binomial, df_long, weight = count)
summary(fit)
r = 1-1/(1+exp(-fit$coefficients)) # retention rate
```
General Retention Model (r_t)
```r
fit = glm(churn ~ factor(t), binomial, df_long, count)
summary(fit)
```

#### Migration Model
to be finished

## Clustering [Python](./python_notes.md#clustering-r)

<div align="right">
    <b><a href="#table-of-contents">↥ back to top</a></b>
</div>

### KMeans Clustering [Python](./python_notes.md#kmeans-clustering-r)

```r
fit = kmeans(df,3,100,100) # 3 clusters, 100 max iterations, 100 initializations and k-means choose the best one
```

summary function
```r
summary.kmeans = function(fit)
{
p = ncol(fit$centers)
K = nrow(fit$centers)
n = sum(fit$size)
xbar = t(fit$centers)%*%fit$size/n
print(data.frame(
n=c(fit$size, n),
Pct=(round(c(fit$size, n)/n,2)),
round(rbind(fit$centers, t(xbar)), 2),
RMSE = round(sqrt(c(fit$withinss/(p*(fit$size-1)), fit$tot.withinss/(p*(n-K)))), 4)
))
cat("SSE=", fit$tot.withinss, "; SSB=", fit$betweenss, "; SST=", fit$totss, "\n")
cat("R-Squared = ", fit$betweenss/fit$totss, "\n")
cat("Pseudo F = ", (fit$betweenss/(K-1))/(fit$tot.withinss/(n-K)), "\n\n");
invisible(list(Rsqr=fit$betweenss/fit$totss,
F=(fit$betweenss/(K-1))/(fit$tot.withinss/(n-K))) )
}
```

plot the clusters (for profiling)
```r
plot.kmeans = function(fit,boxplot=F)
{
require(lattice)
p = ncol(fit$centers)
k = nrow(fit$centers)
plotdat = data.frame(
mu=as.vector(fit$centers),
clus=factor(rep(1:k, p)),
var=factor( 0:(p*k-1) %/% k, labels=colnames(fit$centers))
)
print(dotplot(var~mu|clus, data=plotdat,
panel=function(...){
panel.dotplot(...)
panel.abline(v=0, lwd=.1)
},
layout=c(k,1),
xlab="Cluster Mean"
))
invisible(plotdat)
}
```

### Hierarchical Clustering [Python](./python_notes.md#hierarchical-clustering-r)

```r
fit = hclust(df, method = c("average","single","complete","ward.D") )
# average: average distance between points in cluster and new point
# single: minimun distance between points in cluster and new point (chaining)
# complete: maximum distance between points in cluster and new point
# ward.D: minimize within-cluster SSE

plot(fit) # plot the hierarchical tree
fit$merge # a table of merging process; each row is a merge; negative means a point, positive means a cluster
```

## Deep Learning

<div align="right">
    <b><a href="#table-of-contents">↥ back to top</a></b>
</div>

### Neural Networks
Standardization and Normalization
```r
# standardize predictors
df[1:k] = sapply(df[1:k], function(x) (x-mean(x))/sd(x)) # suppose column 1 to k are predictors

# normalize output, if the output layer is sigmoid
df[k+1] = (df[k+1] - min(df[k+1])) / (max(df[k+1]) - min(df[k+1]))
```

simple NN (one hidden layer)
```r
library(nnet)
nn = nnet(target ~ ., df, linout=T, skip=F, size=10, decay=0.1, maxit = 1000, trace=F)
# linout: whether the output is linear
# skip=T adds a linear combination term to model output (originally the output might has a nonlinear activation)
# size: number of nodes in the hidden layer
# decay: regularization multiplier
# maxit: maximum iterations

summary(nn) # this gives all model parameters/weights
yhat = as.numerica(predict(nn1)) # you can add newdata in the predict function
```

### Interpret Neural Networks

#### ALE Plots

first-order ALE
```r
library(ALEPlot)
yhat = function(X.model, newdata) {
 as.numeric(predict(X.model, newdata))
}

par(mfrow=c(2,4)) # 2 rows, each row has 4 plots, modify this based on real df
for (j in 1:8)  {
 ALEPlot(df[,1:8], nn, pred.fun=yhat, J=j, K=50, NA.plot = TRUE)  #K=number of points/bins you want to use for x-axis
 rug(CRT1[,j])
}  ## This creates main effect ALE plots for all k predictor, the black ticks on the x-axis appear when there's a training value.
```

second-order ALE
```r
par(mfrow=c(2,2))  ## This creates 2nd-order interaction ALE plots for x1, x2, x8
ALEPlot(df[,1:8], nn, pred.fun=yhat, J=c(1,2), K=50, NA.plot = TRUE)
ALEPlot(df[,1:8], nn, pred.fun=yhat, J=c(1,8), K=50, NA.plot = TRUE)
ALEPlot(df[,1:8], nn, pred.fun=yhat, J=c(2,8), K=50, NA.plot = TRUE)
```

#### PD Plots
```r
# interaction between x1 and x8
f0 = mean(df$target)
f1 = ALEPlot(df[,1:8], nn, pred.fun=yhat, J=1)
f8 = ALEPlot(df[,1:8], nn, pred.fun=yhat, J=8)
f18 = ALEPlot(df[,1:8], nn, pred.fun=yhat, J=c(1,8))
f18.combined = f0 + outer(f1$f.values, rep(1,13)) + outer(rep(1,51), f8$f.values) + f18$f.values
image(f1$x.values, f8$x.values, f18.combined, xlab=names(df)[1], ylab=names(df)[8], xlim=range(f1$x.values), y=range(f8$x.values), xaxs='i', yaxs='i')
```
