## Table of contents

* [ML Modeling](#ml-modeling)
  * [Logistic Regression](#logistic-regression)
  * [Discriminant Analysis](#discriminant-analysis)
  * [GLMs](#glms)
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

