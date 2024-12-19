# Load necessary libraries
library(tidyverse)
library(caret)
library(rpart)  # For regression trees
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

# Model Training: Regression Tree with Tuning
# Tune the complexity parameter (cp) of the tree
control <- trainControl(method = "cv", number = 10)  # 10-fold cross-validation

# Train model using caret's train function
tune_grid <- expand.grid(cp = seq(0.01, 0.1, by = 0.01))  # Complexity parameter grid
regression_tree_model <- train(y_train ~ ., data = cbind(y_train, X_train), method = "rpart", 
                               trControl = control, tuneGrid = tune_grid)

# Predict on validation data and calculate RMSE
y_pred_val <- predict(regression_tree_model, newdata = X_val)
rmse <- sqrt(mean((y_val - y_pred_val) ^ 2))
cat('Validation RMSE:', rmse, '\n')

# Prepare Scoring Data
# Preprocess scoring data similarly
scoring_data[factor_vars] <- lapply(scoring_data[factor_vars], as.factor)

# Generate Predictions for Scoring Data
predictions <- predict(regression_tree_model, newdata = scoring_data)

# Create Submission DataFrame
submission_file <- data.frame(id = scoring_data$id, CTR = predictions)

# Write the submission file
write.csv(submission_file, 'C:/Users/harsh/OneDrive/Desktop/R project/submission.csv', row.names = FALSE)

cat('Submission file created successfully!\n')
