## Table of contents
* [Data Manipulation](#data-manipulation)
* [Data Visualization](#data-visualization)
* [ML Modeling](#ml-modeling)
  * [Basic Statistics](#basic-statistics)
  * [Linear Regression](#linear-regression)
  * [Logistic Regression](#logistic-regression)
  * [Discriminant Analysis](#discriminant-analysis)
  * [GLMs](#glms)
## Data Manipulation
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
## Data Visualization
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
## ML Modeling
### Basic Statistics
```r
sd(list) # standard deviation
cor(df) # correlation matrix
cor.test(df$v1, df$v2) # correlation test
```
### Linear Regression
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

### GLMs

