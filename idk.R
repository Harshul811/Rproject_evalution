# Load necessary libraries
library(tidyverse)
library(caret)
library(mice)    # For imputation
library(xgboost) # For gradient boosting with xgboost

# Set seed for reproducibility
set.seed(123)

# Load data
analysis_data <- read.csv("C:/Users/harsh/OneDrive/Desktop/R project/analysis_data.csv")
scoring_data <- read.csv("C:/Users/harsh/OneDrive/Desktop/R project/scoring_data.csv")

# Impute missing values in both analysis and scoring data
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
set.seed(125)
train_index <- createDataPartition(y, p = 0.8, list = FALSE)
X_train <- X_matrix[train_index, ]
y_train <- y[train_index]
X_val <- X_matrix[-train_index, ]
y_val <- y[-train_index]

# Convert data to DMatrix format for xgboost
dtrain <- xgb.DMatrix(data = X_train, label = y_train)
dval <- xgb.DMatrix(data = X_val, label = y_val)

# XGBoost parameters
params <- list(
  objective = "reg:squarederror",
  max_depth = 6,
  eta = 0.1,
  subsample = 0.8,
  colsample_bytree = 0.8
)

# Train the model with early stopping (stop if validation RMSE doesn't improve for 100 rounds)
xgb_model <- xgb.train(
  params = params,
  data = dtrain,
  nrounds = 1000,  # Set a large number of rounds to allow early stopping to kick in
  watchlist = list(val = dval),
  early_stopping_rounds = 100,  # Stop training if validation RMSE doesn't improve in 100 rounds
  print_every_n = 10
)

# Predict on validation data and calculate RMSE
pred_val <- predict(xgb_model, newdata = dval)
rmse_val_xgb <- sqrt(mean((pred_val - y_val)^2))
cat('Validation RMSE for XGBoost Model:', rmse_val_xgb, '\n')

# Prepare Scoring Data and make predictions
dscore <- xgb.DMatrix(data = scoring_matrix)
predictions <- predict(xgb_model, newdata = dscore)

# Create submission file
submission_file <- data.frame(id = imputed_scoring_data$id, CTR = predictions)
write.csv(submission_file, 'C:/Users/harsh/OneDrive/Desktop/R project/xgboost_submission.csv', row.names = FALSE)

cat('XGBoost submission file created successfully!\n')
