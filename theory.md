# Statistics
**1. What is correlation?**

Correlation is a measure of linear relationship between two variables. It is defined by the formula:

<img src="https://latex.codecogs.com/gif.latex?\rho_{X,Y}=Corr(X,Y)=\frac{Cov(X,Y)}{\sigma_X\sigma_Y}=\frac{E(X-\mu_X)(Y-\mu_Y)}{\sigma_X\sigma_Y}" /> 

It is a number between -1 and 1, and the higher the number is in absolute value, the stronger the relationship is. 

_Exaple_:
Correlation between housing price and number of rooms is expected to be positive. The more number of rooms a house has, the higher its price is expected, and vice versa.

**2. What is bias-variance trade-off?**

![Bias-Variance Trade-off](https://miro.medium.com/max/1050/1*9hPX9pAO3jqLrzt0IE3JzA.png)

When building a model, we always face the trade-off between a model's bias and variance. A model with low bias means that it is very accurate in its predictions, but it is likely to have high variance, meaning that a small change in your data will lead to a big change in your model predictions, which is what we call over-fitting. On the other hand, a model with high bias and low variance is also not satisfactory because it does not see the pattern of the data you are modeling. A good model is in between those two cases, in which case the model performance won't change a lot when using different test data (low-variance), but still gives a good prediction in general (low-bias).

**3. What is sampling?**

# Data Manipulation
**1. What are some ways of data imputation?**

For time series:

1. Use the previous / next value of the two to impute.
2. Linear interpolation (draw a strict line based on previous and next value and calculate the missing value in the middle at time t)

For tabular dataframe:

1. Random sampling
2. Average imputation
3. Mode imputation (discrete numerical values/categorical variables)
4. KNN imputation (averaging nearest k neighbors)
5. MICE imputation (default: with PMM - predicative mean matching)

# Modeling
**1. What are the differences bwteeen KNN and K-Means Clustering?**

## Linear Regression
TBD
## Logistic Regression
TBD
## GLMs
Check the [Useful Tutorial](https://www.youtube.com/watch?v=vpKpFMUMaVw) here.

## Model evaluation
### Regression
1. Metrics for evaluation
### Classification
1. Confusion matrix
