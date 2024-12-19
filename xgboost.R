# Load necessary libraries
library(tidyverse)
library(caret)
library(rpart)  # For regression trees
library(mice)   # For imputation
library(xgboost) # For XGBoost

# Load data
analysis_data <- read.csv("C:/Users/harsh/OneDrive/Desktop/R project/analysis_data.csv")
scoring_data <- read.csv("C:/Users/harsh/OneDrive/Desktop/R project/scoring_data.csv")

# Impute missing values in the analysis data
imputed_analysis_data <- mice(analysis_data, m = 1, method = 'pmm', maxit = 5)
analysis_data <- complete(imputed_analysis_data)

# Impute missing values in the scoring data
imputed_scoring_data <- mice(scoring_data, m = 1, method = 'pmm', maxit = 5)
scoring_data <- complete(imputed_scoring_data)

# Data Preprocessing
# Convert categorical variables to factors
factor_vars <- c("position_on_page", "ad_format", "age_group", "gender", 
                 "location", "time_of_day", "day_of_week", "device_type")
analysis_data[factor_vars] <- lapply(analysis_data[factor_vars], as.factor)

# Split data into features and target
X <- analysis_data %>% select(-id, -CTR)
y <- analysis_data$CTR

# Train/Test Split (for evaluation purposes)
set.seed(42)
train_index <- createDataPartition(y, p = 0.8, list = FALSE)
X_train <- X[train_index, ]
y_train <- y[train_index]
X_val <- X[-train_index, ]
y_val <- y[-train_index]

# Convert data to matrix format for XGBoost
X_train_matrix <- model.matrix(~ . - 1, data = X_train)  # Create dummy variables
X_val_matrix <- model.matrix(~ . - 1, data = X_val)

# Check dimensions
cat("Dimensions of training features:", dim(X_train_matrix), "\n")
cat("Length of training labels:", length(y_train), "\n")

# Model Training: XGBoost
xgb_model <- xgboost(data = X_train_matrix, label = y_train, 
                     nrounds = 100, 
                     objective = "reg:squarederror", 
                     eval_metric = "rmse", 
                     verbose = 0)

# Predict on validation data and calculate RMSE
y_pred_val <- predict(xgb_model, newdata = X_val_matrix)
rmse <- sqrt(mean((y_val - y_pred_val) ^ 2))
cat('Validation RMSE:', rmse, '\n')

# Prepare Scoring Data
# Preprocess scoring data similarly
scoring_data[factor_vars] <- lapply(scoring_data[factor_vars], as.factor)
scoring_data_matrix <- model.matrix(~ . - 1, data = scoring_data)

# Generate Predictions for Scoring Data
predictions <- predict(xgb_model, newdata = scoring_data_matrix)

# Create Submission DataFrame
submission_file <- data.frame(id = scoring_data$id, CTR = predictions)

# Write the submission file
write.csv(submission_file, 'C:/Users/harsh/OneDrive/Desktop/R project/submission.csv', row.names = FALSE)

cat('Submission file created successfully!\n')
