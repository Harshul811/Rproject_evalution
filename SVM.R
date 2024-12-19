# Load necessary libraries
library(tidyverse)
library(caret)
library(e1071)  # For SVM
library(mice)   # For imputation

# Load data
analysis_data <- read.csv("C:/Users/harsh/OneDrive/Desktop/R project/analysis_data.csv")
scoring_data <- read.csv("C:/Users/harsh/OneDrive/Desktop/R project/scoring_data.csv")

# Check for NA values before imputation
cat('NA values in analysis_data before imputation:', sum(is.na(analysis_data)), '\n')
cat('NA values in scoring_data before imputation:', sum(is.na(scoring_data)), '\n')

# Impute missing values in the analysis data
imputed_analysis_data <- mice(analysis_data, m = 1, method = 'pmm', maxit = 5)
analysis_data <- complete(imputed_analysis_data)

# Check for NA values after imputation
cat('NA values in analysis_data after imputation:', sum(is.na(analysis_data)), '\n')

# Impute missing values in the scoring data
imputed_scoring_data <- mice(scoring_data, m = 1, method = 'pmm', maxit = 5)
scoring_data <- complete(imputed_scoring_data)

# Check for NA values after imputation
cat('NA values in scoring_data after imputation:', sum(is.na(scoring_data)), '\n')

# Data Preprocessing
# Convert categorical variables to factors
analysis_data$position_on_page <- as.factor(analysis_data$position_on_page)
analysis_data$ad_format <- as.factor(analysis_data$ad_format)
analysis_data$age_group <- as.factor(analysis_data$age_group)
analysis_data$gender <- as.factor(analysis_data$gender)
analysis_data$location <- as.factor(analysis_data$location)
analysis_data$time_of_day <- as.factor(analysis_data$time_of_day)
analysis_data$day_of_week <- as.factor(analysis_data$day_of_week)
analysis_data$device_type <- as.factor(analysis_data$device_type)

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

# Model Training: Support Vector Machine
svm_model <- svm(y_train ~ ., data = data.frame(y_train, X_train), type = 'eps-regression')

# Predictions on validation data
y_pred_val <- predict(svm_model, newdata = data.frame(X_val))

# Calculate RMSE for validation predictions
rmse <- sqrt(mean((y_val - y_pred_val) ^ 2))
cat('Validation RMSE:', rmse, '\n')

# Preprocess scoring data similarly
scoring_data$position_on_page <- as.factor(scoring_data$position_on_page)
scoring_data$ad_format <- as.factor(scoring_data$ad_format)
scoring_data$age_group <- as.factor(scoring_data$age_group)
scoring_data$gender <- as.factor(scoring_data$gender)
scoring_data$location <- as.factor(scoring_data$location)
scoring_data$time_of_day <- as.factor(scoring_data$time_of_day)
scoring_data$day_of_week <- as.factor(scoring_data$day_of_week)
scoring_data$device_type <- as.factor(scoring_data$device_type)

# Generate Predictions for Scoring Data
predictions <- predict(svm_model, newdata = scoring_data)

# Check the number of rows in scoring_data and predictions
cat('Number of rows in scoring_data:', nrow(scoring_data), '\n')
cat('Number of predictions generated:', length(predictions), '\n')

# Ensure predictions and scoring_data have the same length
if (nrow(scoring_data) == length(predictions)) {
  # Create Submission DataFrame
  submission_file <- data.frame(id = scoring_data$id, CTR = predictions)
  
  # Write the submission file
  write.csv(submission_file, 'C:/Users/harsh/OneDrive/Desktop/R project/submission.csv', row.names = FALSE)
  
  cat('Submission file created successfully!\n')
} else {
  cat('Error: Number of predictions does not match number of rows in scoring_data.\n')
}
