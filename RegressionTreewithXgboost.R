# Load necessary libraries
library(tidyverse)
library(caret)
library(mice)   # For imputation
library(gbm)    # For gradient boosting

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

# Combine features and target for gbm
train <- data.frame(CTR = y_train, X_train)

# Set seed for reproducibility
set.seed(123)

# Fit the Gradient Boosting Model
boost <- gbm(CTR ~ .,
             data = train,
             distribution = "gaussian",
             n.trees = 500,
             interaction.depth = 2,
             shrinkage = 0.01)

# Predict on training data and calculate RMSE
pred_train <- predict(boost, newdata = train, n.trees = 500)
rmse_train_boost <- sqrt(mean((pred_train - train$CTR)^2))
cat('Training RMSE for Boosting Model:', rmse_train_boost, '\n')

# Prepare Validation Data
val <- data.frame(CTR = y_val, X_val)

# Predict on validation data and calculate RMSE
pred_val <- predict(boost, newdata = val, n.trees = 500)
rmse_boost <- sqrt(mean((pred_val - val$CTR)^2))
cat('Validation RMSE for Boosting Model:', rmse_boost, '\n')

# Prepare Scoring Data
# Preprocess scoring data similarly
scoring_data[factor_vars] <- lapply(scoring_data[factor_vars], as.factor)
scoring_data_matrix <- data.frame(scoring_data)

# Generate Predictions for Scoring Data
predictions <- predict(boost, newdata = scoring_data_matrix, n.trees = 500)

# Create Submission DataFrame
submission_file <- data.frame(id = scoring_data$id, CTR = predictions)

# Write the submission file
write.csv(submission_file, 'C:/Users/harsh/OneDrive/Desktop/R project/xboost.csv', row.names = FALSE)

cat('Submission file created successfully!\n')
