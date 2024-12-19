# Load necessary libraries
library(tidyverse)
library(caret)
library(mice)    # For imputation
library(rpart)   # For Decision Tree
library(xgboost) # For XGBoost

# Set the seed to 125 (as already known)
set.seed(125)

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
train_index <- createDataPartition(y, p = 0.8, list = FALSE)
X_train <- X_matrix[train_index, ]
y_train <- y[train_index]
X_val <- X_matrix[-train_index, ]
y_val <- y[-train_index]

# 1. Final Decision Tree Model
dt_model <- rpart(y_train ~ ., data = as.data.frame(X_train), method = "anova")

# Predict on validation data and calculate RMSE for Decision Tree
pred_val_dt <- predict(dt_model, as.data.frame(X_val))
rmse_val_dt <- sqrt(mean((pred_val_dt - y_val)^2))
cat("Decision Tree RMSE on validation set:", rmse_val_dt, "\n")

# Predict on scoring data
predictions_dt <- predict(dt_model, as.data.frame(scoring_matrix))

# Create Submission DataFrame for Decision Tree
submission_file_dt <- data.frame(id = imputed_scoring_data$id, CTR = predictions_dt)
write.csv(submission_file_dt, 'C:/Users/harsh/OneDrive/Desktop/R project/dt_125.csv', row.names = FALSE)
cat('Decision Tree submission file created successfully!\n')

# 2. Final XGBoost Model
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

# Train the XGBoost model with early stopping
xgb_model <- xgb.train(
  params = params,
  data = dtrain,
  nrounds = 1000,
  watchlist = list(val = dval),
  early_stopping_rounds = 100,
  print_every_n = 10
)

# Predict on validation data and calculate RMSE for XGBoost
pred_val_xgb <- predict(xgb_model, newdata = dval)
rmse_val_xgb <- sqrt(mean((pred_val_xgb - y_val)^2))
cat("XGBoost RMSE on validation set:", rmse_val_xgb, "\n")

# Predict on scoring data
scoring_data_matrix <- xgb.DMatrix(data = scoring_matrix)
predictions_xgb <- predict(xgb_model, newdata = scoring_data_matrix)

# Create Submission DataFrame for XGBoost
submission_file_xgb <- data.frame(id = imputed_scoring_data$id, CTR = predictions_xgb)
write.csv(submission_file_xgb, 'C:/Users/harsh/OneDrive/Desktop/R project/xgb_dt_125_1.csv', row.names = FALSE)
cat('XGBoost submission file created successfully!\n')

# 3. Linear Model and Backward Stepwise Selection

# Assuming `train` is your training dataset with the relevant target variable (e.g., CTR)
train_data <- as.data.frame(cbind(X_train, CTR = y_train))  # Combine X_train and y_train

# Create a linear model using the training data
start_mod = lm(CTR ~ ., data = train_data)

# Create empty and full models for backward selection
empty_mod = lm(CTR ~ 1, data = train_data)
full_mod = lm(CTR ~ ., data = train_data)

# Perform backward stepwise selection
backwardStepwise = step(start_mod,
                        scope = list(upper = full_mod, lower = empty_mod),
                        direction = 'backward')

# Display the result of stepwise selection
cat("Selected model after backward stepwise selection: \n")
summary(backwardStepwise)
