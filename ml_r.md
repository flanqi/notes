## Table of contents
* [Data Manipulation](#data-manipulation-python)
* [Data Visualization](#data-visualization-python)
* [Basic Supervised Learning](#basic-supervised-learning-python)
  * [Basic Statistics](#basic-statistics)
  * [Linear Regression](#linear-regression-python)
  * [Logistic Regression](#logistic-regression-python)
  * [GLMs](#glms)
  * [Nonlinear Regression](#nonlinear-regression)
  * [Discriminant Analysis](#discriminant-analysis)
  * [Survival Analysis](#survival-analysis)
* [Deep Learning](#deep-learning)
* [Trees](#trees)
    * [Gradient Boosting Trees](#gradient-boosting-trees)
    * [Random Forest](#random-forest)
* [Time Series](#time-series)
* [Nonparametric Methods](#nonparametric-methods)
    * [KNN](#knn)
    * [LOESS](#loess)
    * [GAM](#gam)
    * [PPR](#ppr)
* [Unsupervised Learning](#unsupervised-learning) 
  * [Clustering](#clustering-python)
    * [KMeans Clustering](#kmeans-clustering-python)
    * [Hierarchical Clustering](#hierarchical-clustering-python)
    * [Gaussian Mixture Model](#gaussian-mixture-python)
  * [Dimension Reduction](#dimension-reduction)
    * [Principal Componenet Analysis](#pca) 
    * [Factor Analysis](#factor-analysis)


## Data Manipulation

<div align="right">
    <b><a href="./ml_python.md#data-manipulation-r">⇄ python | </a></b>
    <b><a href="#table-of-contents">↥ back to top</a></b>
</div>

### Read data
```r
df <- read.csv("data.csv", skip = 1, stringAsFactors = F) # skip first row of data
```
### Data Types
```r
class(object) # returns the datatype of the object

df$col <- parse_number(df$col) # parse the column type to numberm, needs tidyverse library

# datetime format
df$date <- as.POSIXct(df$date, format = '%Y%m%d:%H:%M:%S')
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
cols <- c(paste0('v',1:3))
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
df <- iris
df$Sepal.Length <- NULL # this doesn't change original df
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
df <- na.omit(df) # drop all rows with nas
df[complete.cases(df), ] # alternative
```
### Substrings
```r
substr("hello",1,4) # -> "hell"
substr(df$Species,1,4) 
```
## Data Visualization

<div align="right">
    <b><a href="./ml_python.md#data-visualization-r">⇄ python | </a></b>
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

## Basic Supervised Learning

<div align="right">
    <b><a href="./ml_python.md#basic-ml-r">⇄ python | </a></b>
    <b><a href="#table-of-contents">↥ back to top</a></b>
</div>

### Basic Statistics
```r
sd(list) # standard deviation
cor(df) # correlation matrix
cor.test(df$v1, df$v2) # correlation test
```
### Linear Regression
<div align="right">
    <b><a href="./ml_python.md#linear-regression-r">⇄ python</a></b>
</div>

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
ridgecv <- cv.glmnet(x, y, alpha=0,lambda=seq(0,5,0.001),nfold=3) # lasso alpha = 1
ridgecv$lambda.min # best lambda
```

```r
small.lambda.index <- which(ridgecv$lambda == ridgecv$lambda.min)
small.lambda.betas <- coef(ridgecv$glmnet.fit)[,small.lambda.index]
print(small.lambda.betas)
```

### Logistic Regression
<div align="right">
    <b><a href="./ml_python.md#logistic-regression-r">⇄ python</a></b>
</div>

#### Binary Logistic Regression
```r
# fit the model
model <- glm(Y ~ X, family = binomial, data = df) 

# prediction
predict <- predict(model, newdata = test, type = "response") 

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
prob <- c()
CCRs <- c() # choose your own metric, here I use CCR/accuracy

for (p in seq(0,1,0.01)){
  prob <- c(prob,p)
  tab <- table(df$y, predict(model, df, type = "response")>p)
  CCR <- sum(diag(tab))/sum(tab)
  CCRs <- c(CCRs, CCR)
}

p_star <- prob[which.max(CCRs)] # the optimal p

tab = table(radiation$y, probs>p_star) # the final confusion matrix

# other metrics
sensitivity <- tab[2,2]/sum(tab[2,])
specificity <- tab[1,1]/sum(tab[1,])
precision <- tab[2,2]/sum(tab[,2])
recall <- sensitivity
f1 <- 1/(0.5*((1/precision)+(1/recall)))

cbind('sensitivity'=sensitivity, 'specificity' = specificity,
      'F1' = f1)
```

**Imbalance Correction with Bayes Classifier**

In binary classification, sometimes we have imbalanced data (p<0.1), and in this case, we want to upsample our minority class so that 0.3 <= p_s <= 0.7 (p_s is the corrected probability). (If CV is used, make sure to upsample within CV, i.e., CV before upsampling).

Steps for Imbalance Correction:
1. Calculate <img src="https://latex.codecogs.com/svg.latex?p=P(Y=1)" /> for the entire original training set.
2. Balance the training data to have the desired fraction <img src="https://latex.codecogs.com/svg.latex?p_s" />.
3. Caculate the population odds <img src="https://latex.codecogs.com/svg.latex?O=\frac{p}{1-p}" /> and <img src="https://latex.codecogs.com/svg.latex?O_s=\frac{p_s}{1-p_s}" /> for the original and balanced samples, respectively.
4. Fit your classification model <img src="https://latex.codecogs.com/svg.latex?p_s(\mathbf{x})=P[Y=1|\mathbf{x}]" /> to the balanced data.
5. Recover the corrected classification model as follows:
<p align="center">
<img src="https://latex.codecogs.com/svg.latex?p(\mathbf{x})=P[Y=1|\mathbf{x}]=\frac{p_s(\mathbf{x})O}{O_s-p_s(\mathbf{x}(O_s-O))}" />
</p>

#### Nominal Logistic Regression
```r
library(nnet)

# reset the reference group
df$Y <- factor(df$Y)
df$Y <- relevel(df$Y, reference_level)

# fit the model
model <- multinom(Y ~ X, data = df, maxit = 1000) 

# prediction probabilities
predict <- predict(model, newdata = test, type = "probs") 
# predict <- fitted(model, outcome=F) # predictions for training data

# prediction for groups
pred_class <- rep(0,nrow(test))

for(i in 1:nrow(test)){
  pred_class[i] <- which.is.max(predict[i,]) # this outputs the col number with max prob
}
pred_class
```

Confusion Matrix:
```r
library(caret)
cfm <- confusionMatrix(factor(test$Y), factor(pred_class))
cfm
```

#### Ordinal Logistic Regression
```r
# fit the model
library(ordinal)
df$Y <- as,ordered(df$Y) # default order
df$Y <- ordered(df$Y, levels = c(A,B,C), labels = c(A,B,C)) # set new order
model <- clm(Y ~ X, data = df) 

# prediction
predict <- predict(model, newdata = test) 
```

### GLMs
to be finished.
### Nonlinear Regression
This section is about parametric nonlinear regression models and nonlinear least squares. In this problem, we have:
<p align="center">
  <img src="https://latex.codecogs.com/svg.latex?Y_i=g(\boldsymbol{x_i},\boldsymbol{\theta})+\epsilon_i" /> 
</p>
, where g is the parametric non-linear function we use to model the relationship between Y and x, and epsilon is assumed to be normal with zero mean.

nlm() package:
```r
x1 <- df$x1; x2 <- df$x2; y <- df$y

fh <- function(p) {
 yhat <- p[1] + p[2]*x1 + p[3]*exp(p[4]*x2) # the nonlinear function
 return(sum((y-yhat)^2) # return SSE
}
out <- nlm(fn, p=c(1,0,-0.5,-1), hessian = T)  # one can use some approximation methods to get a good initialization for p
# hessian is used for deriving Fisher info matrix

theta <- out$estimate # parameter estimates
```
nls() package:
```r
fn <- function(X1,x2,p){p[1] + p[2]*x1 + p[3]*exp(p[4]*x2)}
out <- nls(y~fn(x1,x2,p), start=list(p=(1,0,-0.5,-1), trace=T)
summary(out)
```

Estimating SEs for Parameters:
1. Using Fisher Info Matrix (applicable with alrge sample size n)
```r
# nlm
H <- out$hessian
MSE <- out$minimum/(nrow(df)-p) # p = number of predictors
I <- (1/(2*MSE))*H # estimate of Fisher Info Matrix

Cov <- solve(I)
se <- sqrt(diag(Cov)) # SEs for parameters

# nls
Cov <- vcov(out)
se <- sqrt(diag(Cov))
```

2. Boostrapping

We use bootstrapping when we don't have large sample size to invoke asymptotics for our parameters, in which case we don't know the population distribution of parameters.

```r
library(boot)

datafit <- function(Z,i, theta0){
 Zboot <- Z[i,]
 x <- Zboot[[2]]; y <- Zboot[[1]]
 fn <- function(p){yhat <- p[1]*x/(p[2]+x); sum((y-yhat)^2)} # same as nlm() fn
 out <- nlm(fn, p=theta0)
 theta <- out$estimate
}

databoot <- boot(df, datafit, R=20000, theta0=c(29.6, 13.4)) # R: number of copies of samples

plot(databoot, index=1) # distribution and qq plot of first parameter
# if the DSTN is approx. normal, we may use crude CI

# estimate SEs
Cov <- cov(datafit$t)
se <- sqrt(diag(Cov))

# estimate parameters
databoot$t0

# CIs
boot.ci(databoot, conf = (.95,.99), index = 1, type = c("norm","basic")) # normal means crude CI, basic means reflected CI
```

Calculate CI and PI for a response:
```r
# CI
datafit <- function(Z,i,theta0, x_pred){
 Zboot <- Z[i,]
 x <- Zboot[[2]]; y = Zboot[[1]]
 fn <- function(p){yhat = p[1]*x/(p[2]+x); sum((y-yhat)^2)} # same as nlm() fn
 out <- nlm(fn, p=theta0)
 theta <- out$estimate
 y_pred <- theta[1]*x_pred/(theta[2]+x_pred)
}

databoot <- boot(df, datafit, R=20000, theta0 = c(29.6,13.4), x_pred = 27)
boot.ci(databoot, type = c("norm", "basic")) # CI

# PI
Yhat0 <- databoot$t0
SEYhat <- sqrt(var(databoot$t))
SEY <- sqrt(SEYhat^2 + MSE)

c(Yhat0-qnorm(0.975)*SEY,Yhat0+qnorm(0.975)*SEY) # PI
```

### Discriminant Analysis
#### Fisher Discriminant Function
```r
library(MASS)
iris <- read.csv("Iris.csv")
fit <- lda(Species_name~., data=iris[2:6], prior = c(1/3,1/3,1/3))
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
fit <- survfit(Surv(t, churn) ~ 1. data = df, weight = count) # df is not in long-form, at each customer time t, the number of customers churned / censored are stored in variable "count"
summary(fit)
```

Survival Function Plot (this model gives the same results as in KM model):
```r
plot(fit)
```

survfit doesn't provide harzard and pdf plots, but we can extract results from the summary table.
```r
# table
res <- summary(fit)
cols <- lapply(c(2:6,8:11), function(x) res[x])
table <- do.call(data.frame, cols)

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
fit <- survfit(Surv(t, churn) ~ servicelen, data = df, weight = count)

# survival function plot
plot(fit, lty = 1:3) # suppose we have 3 types of service length 
legend(0.5,0.3,c('1 month', '6 months', '12 months'), lty=1:3, cex=0.8)
```
we can use the same way to extract summary table and plot harzard rates and pdf (in this case, we need to separate the tables into 3 groups).

#### Discrete Time Model

In discrete time model, we need to convert the data into long format, i.e, each (or each group of) customer has one row at each customer time t.

Turn data into long format:
```r
df_long <- survSplit(data = df, cut = 0:12, end = 't', event = 'churn')
# here cut is usually 0 : max(t)
```

Simple Retention Model (r_t = r)
```r
fit <- glm(churn ~ 1, binomial, df_long, weight = count)
summary(fit)
r <- 1-1/(1+exp(-fit$coefficients)) # retention rate
```
General Retention Model (r_t)
```r
fit <- glm(churn ~ factor(t), binomial, df_long, count)
summary(fit)
```

#### Migration Model

Simplest case: Only 2 states, buyers (1) and non-buyers (0). Customer vector n=(n0, n1); Transition matrix P=(p00,p01\\p10,p11); Net revenue vector v=(v0,v1).
General case: more states, taking into account recency and frequency (e.g. new customers who made purchases, old customers who made purchases, customers who hasn't made purchase for 1 period, 2 period, etc.). We could have a net revenue matrix, each column is a different type of revenue, e.g. revenue from subscriptions or ads.

Number of Customers at Period k: t(n) x P^k
Customer Equity: t(n) x (I - P/(1+d))^{-1} x v

```r
# use prop.table(matrix, 1) to create transition matrix
lab <- c("Churn", "Basic", "Premium")
P <- prop.table(matrix(c(10000,400,200, 1800, 300,100, 240,500,450), nrow=3, ncol=3, byrow=T, dimnames=list(lab,lab), 1)

# number of customers in each segment in two months
n <- c(4000, 4500, 1000, 490)
t(n)%*%P%*%P

# revenue vector
rev <- matrix(c(0,1, 10,3, 17,0), nrow=3, dimnames=list(lab, c("Subscription","Ad")))

# CE
d <- 0.01
t(n)%*%solve(diag(3)-P/(1+d))%*%v # outputs CE for subs and id separately
```

## Deep Learning

<div align="right">
    <b><a href="#table-of-contents">↥ back to top</a></b>
</div>

### Neural Networks
Standardization and Normalization
```r
# standardize predictors
df[1:k] <- sapply(df[1:k], function(x) (x-mean(x))/sd(x)) # suppose column 1 to k are predictors

# normalize output, if the output layer is sigmoid
df[k+1] <- (df[k+1] - min(df[k+1])) / (max(df[k+1]) - min(df[k+1]))
```

simple NN (one hidden layer)
```r
library(nnet)
nn <- nnet(target ~ ., df, linout=T, skip=F, size=10, decay=0.1, maxit = 1000, trace=F)
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
yhat <- function(X.model, newdata) {
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
f0 <- mean(df$target)
f1 <- ALEPlot(df[,1:8], nn, pred.fun=yhat, J=1)
f8 <- ALEPlot(df[,1:8], nn, pred.fun=yhat, J=8)
f18 <- ALEPlot(df[,1:8], nn, pred.fun=yhat, J=c(1,8))
f18.combined <- f0 + outer(f1$f.values, rep(1,13)) + outer(rep(1,51), f8$f.values) + f18$f.values
image(f1$x.values, f8$x.values, f18.combined, xlab=names(df)[1], ylab=names(df)[8], xlim=range(f1$x.values), y=range(f8$x.values), xaxs='i', yaxs='i')
```

## Trees

<div align="right">
    <b><a href="#table-of-contents">↥ back to top</a></b>
</div>

### Basic Trees
There is no need to standardize predictors because trees are not influenced by linear transformations.

```r
library(rpart)

control <- rpart.control(minbucket=5, cp=0.0001, xval=10) # 10-fold built-in CV
fit <- rpart(target~., df, method = "anova", control=control)

plotcp(fit) # plot the 1-R^2 cv versus cp
printcp(fit) # find optimal cp based on xerror

# pruning
fit <- prune(fit, cp=optimal_cp)
fit$cptable[nrow(model4$cptable),] # summary

# feature importance (based on reduction in SSE)
fit$variable.importance

# splits
fit 

# plot the tree
library(rpart.plot)
rpart.plot(fit)
```

The code is the same for classification trees. You only need to change the method to "class", and factor your target variable. Notice that the xerror in this case will be cv misclassifcation rate / stump misclassifcation rate. Follow the code below to recover cv misclassification rate:
```r
tbl <- table(df$type)/nrow(df) # type is the target class
optimal_xerror*(1-max(tbl))
```

### Gradient Boosting Trees

```r
library(gbm)
set.seed(1)
gbm <- gbm(y~, data=df, distribution="gaussian", n.trees=1000, shrinkage=0.01, interaction.depth=3, cv.folds=nrow(heartdisease),verbose=F)

best.iter <- gbm.perf(gbm, method="cv"); best.iter
1-gbm$cv.error[best.iter]/var(heartdisease$cost)
summary(gbm)

library(gridExtra)
plotlist <- vector(mode='list', length=8)
for (i in 1:8){
 plotlist[[i]] <- plot(gbm, i.var=i, n.trees=best.iter)
}
do.call(grid.arrange, c(plotlist,nrow=2,ncol=4))

predict(gbm, newdata, n.trees=best.iter)
```

### Random Forest

```r
library(randomForest)

set.seed(1)
rf <- randomForest(y~., data=df, mtry=3, ntree=500, nodesize=3, importance=T)

plot(rf)
print(rf)

importance(rf); varImpPlot(rf)

par(mfrow=c(2,4))
for (i in c(2:9)){
 partialPlot(rf, pred.data=df, x.var=names(df)[i], xlab=names(heartdisease)[i], main=NULL)
}

predict(rf, newdata)
r2 <- 1-rf$mse[rf$ntree]/var(df$y) # use OOB MSE to compute OOB R^2
```

## Time Series

<div align="right">
    <b><a href="#table-of-contents">↥ back to top</a></b>
</div>

### Moving Average 
MA is usually used for smoothing (out seasonality).

```r
y = ts(df[[1]], deltat=1/12)
m <- 12; n <- length(y)
MA <- filter(y, filter=rep(1/m,m), method="convolution", sides=2) # centered smoothing
plot(y, type="b")
lines(MA, col="red")
```

### Exponentially Weighted Moving Average
Only level prediction.
```r
y <- ts(df[[1]], deltat=1/12)
k <- 12 # prediction window
EWMA <- HoltWinters(y, seasonal="additive", beta=FALSE, gamma=FALSE)
EWMAPred <- predict(EWMA, n.ahead=k, prediction.interval=T, level=.95)
plot(EWMA, EWMAPred type="b")
EWMA
```
### Holt Method
Only level and trend prediction. 
```r
Holt <- HoltWinters(y, seasonal="additive", gamma=FALSE)
HoltPred <- predict(Holt, n.ahead=k, prediction.interval=T, level=.95)
plot(Holt, HoltPred type="b")
Holt
```

### Holt Winters Method
Level, trend and seasonality prediction. 
```r
HW <- HoltWinters(y, seasonal="additive")
HWPred <- predict(HW, n.ahead=k, prediction.interval=T, level=.95)
plot(HW, HW, type="b")
HW
```

If the amplitude of each seasonlity depends on trend (e.g. increasing amplitude and increasing trend), then a multiplicative method should perform better (set seasonal="multiplicative").

### Time Series Decomposition

We decompose ts into 3 parts: trend, seasonality and random errors.
```r
Dec <- decompose(y. type="additive")
plot(Dec, type="b")

Dec
```

Prediction:
```r
y_hat <- Dec$trend + Dec$seasonal # Dec$trend*Dec$seasonal for mult. method
plot(y, type="b")
lines(y_hat, col="red")
```

## Nonparametric Methods

<div align="right">
    <b><a href="#table-of-contents">↥ back to top</a></b>
</div>

### KNN 

```r
library(yaImpute)

# need standardization beforehand
# use CV to choose the best K

train <- as.matrix(df[,2:9]); ytrain = df[,1]
test <- as.matrix(df[,2:9]) # same as train if you want training performance

out <- ann(train, test, K, verbose=F) # K=number of neighbors
ind <- as.matrix(out$knnIndexDist[,1:K]) # the k-neighbors' indices for each observation

fit <- apply(ind, 1, function(x) mean(ytrain[x])) # training predictions
# fit <- apply(ind, 1, function(x) sum(ytrain[x]==1/length(ytrain[x])) for classification
```

### LOESS

```r
# std predictors first
out <- loess(y~., df, degree=1, span=.2) # degree=1:linear, degree=2:quadratic; bigger span means widder window
predict <- predict(out, newdata)
```

Use Cp to choose parameters:
```r
for deg in c(0,1,2){
 for (lambda in seq(0.1,0.5, 0.05)){
  out <- loess(y~.df, degree=deg, span=lambda)
  SSE <- sum((df$y-out$fitted)^2)
  Cp <- (SSE+2*out$trace.hat*(out$s^2))/nrow(df)
  print(c(deg,lambda,Cp))
 }
}
```

### GAM

```r
library(mgcv)

# standardize predictors first
out <- gam(y~s(x1)+s(x2)+x3, data=df, family=gaussian(), sp=c(-1,-1)) # family=binomial() if classification, sp=-1 if uses R to find optimal sp
summary(out)

# pd plots
par(mfrow=c(2,4))
plot(out)
```

### PPR

```r
out <- ppr(y~., data=df, nterms=M) # M=number of basis functions, use CV to tune
summary(out)
plot(out) # pd plots
```

## Unsupervised Learning
### Clustering
<div align="right">
    <b><a href="./ml_python.md#clustering-r">⇄ python | </a></b>
    <b><a href="#table-of-contents">↥ back to top</a></b>
</div>

#### KMeans Clustering
<div align="right">
    <b><a href="./ml_python.md#kmeans-clustering-r">⇄ python</a></b>
</div>
```r
fit <- kmeans(df,3,100,100) # 3 clusters, 100 max iterations, 100 initializations and k-means choose the best one
```

summary function
```r
summary.kmeans <- function(fit)
{
p <- ncol(fit$centers)
K <- nrow(fit$centers)
n <- sum(fit$size)
xbar <- t(fit$centers)%*%fit$size/n
print(data.frame(
n <- c(fit$size, n),
Pct <- (round(c(fit$size, n)/n,2)),
round(rbind(fit$centers, t(xbar)), 2),
RMSE <- round(sqrt(c(fit$withinss/(p*(fit$size-1)), fit$tot.withinss/(p*(n-K)))), 4)
))
cat("SSE=", fit$tot.withinss, "; SSB=", fit$betweenss, "; SST=", fit$totss, "\n")
cat("R-Squared = ", fit$betweenss/fit$totss, "\n")
cat("Pseudo F = ", (fit$betweenss/(K-1))/(fit$tot.withinss/(n-K)), "\n\n");
invisible(list(Rsqr=fit$betweenss/fit$totss,
F <- (fit$betweenss/(K-1))/(fit$tot.withinss/(n-K))) )
}
```

plot the clusters (for profiling)
```r
plot.kmeans = function(fit,boxplot=F)
{
require(lattice)
p <- ncol(fit$centers)
k <- nrow(fit$centers)
plotdat <- data.frame(
mu <- as.vector(fit$centers),
clus <- factor(rep(1:k, p)),
var <- factor( 0:(p*k-1) %/% k, labels=colnames(fit$centers))
)
print(dotplot(var~mu|clus, data=plotdat,
panel <- function(...){
panel.dotplot(...)
panel.abline(v=0, lwd=.1)
},
layout <- c(k,1),
xlab <- "Cluster Mean"
))
invisible(plotdat)
}
```

#### Hierarchical Clustering
<div align="right">
    <b><a href="./ml_python.md#hierarchical-clustering-r">⇄ python</a></b>
</div>
##### Distance Metrics 

Numerical Data:
```r
library(proxy)
x <- matrix(c(5,1,4,1), nrow=2) # matrix (5,4 // 1,1)

# Distance
dist(x) # Euclidean distance of 2 rows
dist(x, method="maximum") # 4
dist(x, method="manhattan") # 4+3 = 7

# Similarity
simil(x) # pearson corr; dist(x, method="correlation") = 1-simil(x)
simil(x, method="cosine")
```

Binary Data:
* Jaccard similarity
* Simple Matching similarity (co-absences are informative)
```r
x <- matrix(c(1,1,1,1,0,0,0,0,0,0,0,0,
             0,0,0,1,1,0,1,0,0,0,0,0,
             0,1,1,0,1,0,1,0,1,1,1,0), byrow=T, ncol=12, dimnames=list(LETTERS[1:3]))
simil(x, method="cosine"); 1-dist(x, method="cosine")
simil(x, method="Jaccard")
```

##### hclust
```r
fit <- hclust(df, method = c("average","single","complete","ward.D") )
# average: average distance between points in cluster and new point
# single: minimun distance between points in cluster and new point (chaining)
# complete: maximum distance between points in cluster and new point
# ward.D: minimize within-cluster SSE

plot(fit) # plot the hierarchical tree
fit$merge # a table of merging process; each row is a merge; negative means a point, positive means a cluster
```

#### Gaussian Mixutre
<div align="right">
    <b><a href="./ml_python.md#gaussian-mixture-r">⇄ python</a></b>
</div>
```r
library(mclust) 
fit <- Mclust(data, G=2, modelNames = "VVI") # 2 clusters
# one can leave out the modelNames and plot the fit to see the comparisons of different models
plot(fit) # plot clusters with elipses, BIC plot of different models, etc.

# proportion of clusters
fit$parameters$pro

# mean 
fit$parameters$mean

# covariance matrices
fit$parameters$vairance$sigma

# correlation matrices
cov2cor(fit$parameters$vairance$sigma[,,1]) # of the first cluster
```

### Dimension Reduction

<div align="right">
    <b><a href="#table-of-contents">↥ back to top</a></b>
</div>

### PCA

prcomp package: (uses svd on the dataframe, preferred method)
```r
fit <- prcomp(data, scale=T) # scale = T if use correlation (incommensurate units)
plot(fit) # scree plot
summary(fit) # std, proportion of variance

# loading vector
fit$rotation

# lambda / variances of PCs
fit$sdev^2 # same as var(fit$x)

# fitted values / pc scores
fit$x
```

hand calculation:
```r
cor <- cor(data)
fit <- eigen(cor) 
# $values: variances of PCs / eigenvalues
# $vectors: PCs / eigenvectors

# fitted values / PC scores / projections for each observation
scale(data) %*% fit$vectors
```

princomp package (same as hand calculation, use correlation and eigen, can be supplied only correlation matrix without data):
```r
fit <- princomp(data, cor=T) # standardized if cor=T; use covmat = cor, if cor/cov is supplied instead of dataframe
fit$sdev # std of PCs
fit$loadings # loadings
fit$scores # pc scores
```

principal package (can do rotation of PCs, usually used for factor analysis):
```r
library(psych)
fit <- principal(data, nfactor=2, rotate="none") # need to specify no rotation for PCA and total number of PCs
# standardized data by default

fit$loadings # SS loadings are variances of PCs
# PCs are not normalized, with length equal to std of PC

fit$scores # each PC score is standardized because of scaled PCs 
```

### Factor Analysis

Suppose we have some manifest variables to measure some latent variable, which variables should we include?

Cronbach's coefficient:
```r
library(psych)
alpha(data)
# try dropping some variables and see if it increases alpha
```

There are two ways to estimate latent variables:
* MLE
* PCA

PCA:
```r
fit <- principal(data) # with rotation
fit$loadings 

fit$uniqueness # uniqueness:
1-fit$uniqueness # communality: how much the factors/latent vars account for the variation in x-variables/manifest-vars
```
Each PC/RC is a column vector in the L matrix, V(X) = LL'+V(Error). Let's say the first column vector of L is (4,...), then Cor(X1,f1) = 4 (if X is standardized, the correlation between x1 and the first latent var is 4).

Explanetory Factor Analysis:
We want to find a number of latent variables (e.g. math and verbal ability) by using PCA and rotating axes.

Step 1: Decide the number of factors/latent variables (use scree plot or Kaiser's criterion)
```r
# prcomp or princomp
screeplot(fit)

# principal
plot(1:p, fit$values, type="b") # p = number of dimensions
abline(h=1)
```

Step 2: Perform PCA, and find a more interpretable baiss by rotating the axes:
* Varimax: orthogonal axes with a few large loadings and as many near-zero loadings as possible
* Oblique/Promax: allow correlated factors
* Independent component analysis (ICA): independent factors
```r
fit <- principal(data, nfacor=2, rotate="promax")
fit
```

Step 3: Name and purify factors; drop items with small loadings or substantial cross loadings
```r
print(fit$loadings, cutoff=0.3, digits=3, sort=T)
```

