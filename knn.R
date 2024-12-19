# Install and load necessary packages
# Install lightgbm if you haven't already
install.packages("lightgbm", repos = "https://cran.r-project.org/")

# Load libraries
library(tidyverse)
library(caret)
library(mice)    # For imputation
library(lightgbm) # For LightGBM

# Set the seed for reproducibility
set.seed(125)

# Load data
analysis_data <- read.csv("C:/Users/harsh/OneDrive/Desktop/R project/analysis_data.csv")
scoring_data <- read.csv("C:/Users/harsh/OneDrive/Desktop/R project/scoring_data.csv")

# Impute missing values
imputed_analysis_data <- complete(mice(analysis_data, m = 1, method = 'pmm', maxit = 5))
imputed_scoring_data <- complete(mice(scoring_data, m = 1, method = 'pmm', maxit = 5))

# Convert categorical variables to factors
factor_vars <- c("position_on_page", "ad_format", "age_group", "gender", 
                 "location", "time_of_day", "day_of_week", "device_type")
imputed_analysis_data[factor_vars] <- lapply(imputed_analysis_data[factor_vars], as.factor)
imputed_scoring_data[factor_vars] <- lapply(imputed_scoring_data[factor_vars], as.factor)

# Separate features and target variable for training data
X <- imputed_analysis_data %>% select(-id, -CTR)
y <- imputed_analysis_data$CTR

# Convert categorical features to numeric using dummy variables
dummies <- dummyVars(" ~ .", data = X)
X_matrix <- predict(dummies, newdata = X)
scoring_matrix <- predict(dummies, newdata = imputed_scoring_data)

# Split into training and validation sets
train_index <- createDataPartition(y, p = 0.8, list = FALSE)
X_train <- X_matrix[train_index, ]
y_train <- y[train_index]
X_val <- X_matrix[-train_index, ]
y_val <- y[-train_index]

# Convert training and validation sets into lightgbm datasets
dtrain <- lgb.Dataset(data = X_train, label = y_train)
dval <- lgb.Dataset(data = X_val, label = y_val)

# Set parameters for the LightGBM model
params <- list(
  objective = "regression",         # regression task (since CTR is continuous)
  metric = "rmse",                  # root mean squared error
  boosting_type = "gbdt",            # gradient boosting decision tree
  num_leaves = 31,                  # maximum number of leaves in one tree
  max_depth = -1,                   # no maximum depth
  learning_rate = 0.05,             # step size for updating weights
  nthread = 4,                      # number of threads to use
  min_data_in_leaf = 20,            # minimum number of data points in a leaf
  colsample_bytree = 0.8,           # fraction of features to use for each tree
  subsample = 0.8                   # fraction of samples to use for each tree
)

# Train the LightGBM model
lgb_model <- lgb.train(
  params = params,
  data = dtrain,
  nrounds = 1000,                   # number of boosting rounds
  valids = list(val = dval),
  early_stopping_rounds = 100       # early stopping if no improvement
)

# Make predictions on the validation set
pred_val_lgb <- predict(lgb_model, X_val)

# Calculate RMSE (Root Mean Squared Error) on validation set
rmse_lgb <- sqrt(mean((pred_val_lgb - y_val)^2))
cat("LightGBM RMSE on validation set:", rmse_lgb, "\n")

# Make predictions on scoring data
predictions_lgb <- predict(lgb_model, newdata = scoring_matrix)

# Create Submission DataFrame for LightGBM
submission_file_lgb <- data.frame(id = imputed_scoring_data$id, CTR = predictions_lgb)
write.csv(submission_file_lgb, 'C:/Users/harsh/OneDrive/Desktop/R project/lgb_predictions_125.csv', row.names = FALSE)
cat('LightGBM submission file created successfully!\n')
