# Load necessary libraries
library(tidyverse)
library(caret)
library(mice)    # For imputation

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
y <- ifelse(imputed_analysis_data$CTR > 0.5, 1, 0)  # Convert CTR to binary outcome (e.g., clicked or not)

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

# 1. Logistic Regression Model
log_reg_model <- glm(y_train ~ ., data = as.data.frame(X_train), family = "binomial")

# Predict on validation data and calculate Accuracy for Logistic Regression
pred_val_log_reg <- predict(log_reg_model, newdata = as.data.frame(X_val), type = "response")
pred_val_log_reg_class <- ifelse(pred_val_log_reg > 0.5, 1, 0)  # Convert probabilities to binary outcome

accuracy_val_log_reg <- mean(pred_val_log_reg_class == y_val)
cat("Logistic Regression Accuracy on validation set:", accuracy_val_log_reg, "\n")

# Predict on scoring data
predictions_log_reg <- predict(log_reg_model, newdata = as.data.frame(scoring_matrix), type = "response")
predictions_log_reg_class <- ifelse(predictions_log_reg > 0.5, 1, 0)

# Create Submission DataFrame for Logistic Regression
submission_file_log_reg <- data.frame(id = imputed_scoring_data$id, CTR = predictions_log_reg_class)
write.csv(submission_file_log_reg, 'C:/Users/harsh/OneDrive/Desktop/R project/log_reg_125.csv', row.names = FALSE)
cat('Logistic Regression submission file created successfully!\n')
