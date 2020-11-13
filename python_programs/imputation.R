colSums(is.na(redwine)) # number of nas for each column
sum(rowSums(is.na(redwine))) # total number of samples that have missing values

# Random imputation function:
random.imp = function(df,target_df=df){
  df.obs = df[!is.na(df)] # extract non-nas
  
  missing = is.na(target_df)
  n.missing = sum(missing) # sum of nas
  
  imputed = target_df
  # sample with replacement
  imputed[missing]=sample(df.obs, n.missing, replace=T)
  return(imputed)
}

# Mode imputation function:
Mode = function(x){
  mode = as.numeric(names(sort(table(x),decreasing = T))[1])
  return(mode)
}

mcv.imp = function(df,target=df){
  imputed = target
  
  for(i in 1:ncol(df)){
    missing = is.na(imputed[i])
    imputed[i][missing] = Mode(df[i])
  }
  
  return(imputed)
}

# Average imputation function:
avg.imp = function(df,target=df){
  imputed = target
  
  for(i in 1:ncol(df)){
    missing = is.na(imputed[i])
    imputed[i][missing] = mean(df[[i]], na.rm = T)
  }
  
  return(imputed)
}

# kNN imputation
library(DMwR)

df.imputed = knnImputation(df)

# MICE imputation
library(mice)

df.imputed = complete(mice(df,seed=400,print=F),5)
