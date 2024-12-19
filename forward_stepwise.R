# Load necessary libraries
library(tidyverse)
library(caret)
library(mice)    # For imputation
library(rpart)   # For Decision Tree
library(xgboost) # For XGBoost
library(glmnet)   # For Ridge and Lasso Regression

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

# 1. Linear Regression Model
lr_model <- lm(y_train ~ ., data = as.data.frame(X_train))

# Predict on validation data and calculate RMSE for Linear Regression
pred_val_lr <- predict(lr_model, as.data.frame(X_val))
rmse_val_lr <- sqrt(mean((pred_val_lr - y_val)^2))
cat("Linear Regression RMSE on validation set:", rmse_val_lr, "\n")

# Predict on scoring data
predictions_lr <- predict(lr_model, as.data.frame(scoring_matrix))

# Create Submission DataFrame for Linear Regression
submission_file_lr <- data.frame(id = imputed_scoring_data$id, CTR = predictions_lr)
write.csv(submission_file_lr, 'C:/Users/harsh/OneDrive/Desktop/R project/lr_125.csv', row.names = FALSE)
cat('Linear Regression submission file created successfully!\n')

# 2. Ridge Regression Model (Using glmnet)
ridge_model <- cv.glmnet(X_train, y_train, alpha = 0)

# Predict on validation data and calculate RMSE for Ridge Regression
pred_val_ridge <- predict(ridge_model, X_val, s = "lambda.min")
rmse_val_ridge <- sqrt(mean((pred_val_ridge - y_val)^2))
cat("Ridge Regression RMSE on validation set:", rmse_val_ridge, "\n")

# Predict on scoring data
predictions_ridge <- predict(ridge_model, scoring_matrix, s = "lambda.min")

# Create Submission DataFrame for Ridge Regression
submission_file_ridge <- data.frame(id = imputed_scoring_data$id, CTR = predictions_ridge)
write.csv(submission_file_ridge, 'C:/Users/harsh/OneDrive/Desktop/R project/ridge_125.csv', row.names = FALSE)
cat('Ridge Regression submission file created successfully!\n')

# 3. Lasso Regression Model (Using glmnet)
lasso_model <- cv.glmnet(X_train, y_train, alpha = 1)

# Predict on validation data and calculate RMSE for Lasso Regression
pred_val_lasso <- predict(lasso_model, X_val, s = "lambda.min")
rmse_val_lasso <- sqrt(mean((pred_val_lasso - y_val)^2))
cat("Lasso Regression RMSE on validation set:", rmse_val_lasso, "\n")

# Predict on scoring data
predictions_lasso <- predict(lasso_model, scoring_matrix, s = "lambda.min")

# Create Submission DataFrame for Lasso Regression
submission_file_lasso <- data.frame(id = imputed_scoring_data$id, CTR = predictions_lasso)
write.csv(submission_file_lasso, 'C:/Users/harsh/OneDrive/Desktop/R project/lasso_125.csv', row.names = FALSE)
cat('Lasso Regression submission file created successfully!\n')
