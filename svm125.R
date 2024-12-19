# Load necessary libraries
library(tidyverse)
library(caret)
library(mice)    # For imputation
library(e1071)    # For Support Vector Machine

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

# 1. Support Vector Machine Model (SVM)
svm_model <- svm(y_train ~ ., data = as.data.frame(X_train), type = "eps-regression")

# Predict on validation data and calculate RMSE for SVM
pred_val_svm <- predict(svm_model, as.data.frame(X_val))
rmse_val_svm <- sqrt(mean((pred_val_svm - y_val)^2))
cat("SVM RMSE on validation set:", rmse_val_svm, "\n")

# Predict on scoring data
predictions_svm <- predict(svm_model, as.data.frame(scoring_matrix))

# Create Submission DataFrame for SVM
submission_file_svm <- data.frame(id = imputed_scoring_data$id, CTR = predictions_svm)
write.csv(submission_file_svm, 'C:/Users/harsh/OneDrive/Desktop/R project/svm_125.csv', row.names = FALSE)
cat('SVM submission file created successfully!\n')
