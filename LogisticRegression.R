# Load necessary libraries
library(tidyverse)
library(caret)
library(mice)   # For imputation

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
analysis_data$position_on_page <- as.factor(analysis_data$position_on_page)
analysis_data$ad_format <- as.factor(analysis_data$ad_format)
analysis_data$age_group <- as.factor(analysis_data$age_group)
analysis_data$gender <- as.factor(analysis_data$gender)
analysis_data$location <- as.factor(analysis_data$location)
analysis_data$time_of_day <- as.factor(analysis_data$time_of_day)
analysis_data$day_of_week <- as.factor(analysis_data$day_of_week)
analysis_data$device_type <- as.factor(analysis_data$device_type)

# Convert CTR to a binary outcome (1 if clicked, 0 otherwise)
analysis_data$clicked <- ifelse(analysis_data$CTR > 0.1, 1, 0)  # Adjust threshold as necessary

# Split data into features and target
X <- analysis_data %>% select(-id, -CTR, -clicked)  # Remove CTR and clicked from features
y <- analysis_data$clicked

# Train/Test Split (for evaluation purposes)
set.seed(42)
train_index <- createDataPartition(y, p = 0.8, list = FALSE)
X_train <- X[train_index, ]
y_train <- y[train_index]
X_val <- X[-train_index, ]
y_val <- y[-train_index]

# Model Training: Logistic Regression
logistic_model <- glm(y_train ~ ., data = X_train, family = binomial)

# Predict on validation data
y_pred_prob <- predict(logistic_model, newdata = X_val, type = "response")
y_pred <- ifelse(y_pred_prob > 0.5, 1, 0)  # Adjust threshold if necessary

# Calculate accuracy and other metrics
accuracy <- mean(y_pred == y_val)
cat('Validation Accuracy:', accuracy, '\n')

# Prepare Scoring Data
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
pred_prob <- predict(logistic_model, newdata = scoring_data, type = "response")
predictions <- ifelse(pred_prob > 0.5, 1, 0)  # Adjust threshold if necessary

# Create Submission DataFrame
submission_file <- data.frame(id = scoring_data$id, CTR = pred_prob)

# Write the submission file
write.csv(submission_file, 'C:/Users/harsh/OneDrive/Desktop/R project/submission.csv', row.names = FALSE)

cat('Submission file created successfully!\n')
