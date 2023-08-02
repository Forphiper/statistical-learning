# Load required packages
library(ISLR)
library(tidyverse)
library(ggplot2)
library(dplyr)
library(caret)
library(e1071)
library(glmnet)
library(gbm)
library(rpart)
library(class)

# Read data
setwd(".")
sentiment <- read.csv("processed_data.csv")
dim(sentiment)
head(sentiment)

# Drop rows which contain na
sentiment <- sentiment[complete.cases(sentiment), ]
dim(sentiment)

# Subset predictors and label
predictor_names <- c()
for (model_idx in 1:1) {
  for (score_idx in 1:25) {
    predictor_name <- paste("model", model_idx, "_top", score_idx, sep = "")
    predictor_names <- c(predictor_names, predictor_name)
  }
}

# # Technical indicators
# for (indicator_idx in 1:83) {
#   predictor_name <- paste("indicator", indicator_idx, sep="")
#   predictor_names <- c(predictor_names, predictor_name)
# }
# predictor_names <- c(predictor_names, "open")
# predictor_names <- c(predictor_names, "high")
# predictor_names <- c(predictor_names, "low")
# predictor_names <- c(predictor_names, "close")

# Subset the dataframe
variables <- c(predictor_names, "label")
sentiment <- sentiment[, variables]
dim(sentiment)
head(sentiment)

# Convert the label column to a numeric vector
sentiment$label <- as.numeric(sentiment$label)

# Separate predictors and label
predictors <- model.matrix(label ~ .-1, data=sentiment)
label <- sentiment$label


# # Add previous rows of features to current row
# num_prev_rows <- 2
# for (prev in 1:num_prev_rows) {
#   for (model_idx in 1:1) {
#     for (score_idx in 1:25) {
#       new_col = paste("model", model_idx, "_top", score_idx, "_prev", prev, sep = "")
#       col_to_shift = paste("model", model_idx, "_top", score_idx, sep = "")
#       sentiment[[new_col]] <- c(rep(NA, prev), sentiment[[col_to_shift]][1:(nrow(sentiment)-prev)])
#     }
#   }
# }
# sentiment <- sentiment[complete.cases(sentiment), ]
# dim(sentiment)
# head(sentiment)
# 
# # Add the previous features to the predictor_names
# for (model_idx in 1:1) {
#   for (score_idx in 1:25) {
#     for (prev in 1:num_prev_rows) {
#       predictor_name <- paste("model", model_idx, "_top", score_idx, "_prev", prev, sep = "")
#       predictor_names <- c(predictor_names, predictor_name)
#     }
#   }
# }

# Perform Ridge feature selection
ridge.fit <- glmnet(predictors, label, alpha = 0)
ridge_selected <- coef(ridge.fit, s = cv.glmnet(predictors, label)$lambda.1se)
ridge_selected
coeff_magnitudes <- abs(ridge_selected)[-1]

# Sort the magnitudes in descending order and get the indices of the top features
top_feature_indices <- order(coeff_magnitudes, decreasing = TRUE)[1:10]
selected_predictor_names = predictor_names[top_feature_indices]
selected_predictor_names

# Subset the dataframe with selected features and label
variables <- c(selected_predictor_names, "label")
sentiment <- sentiment[, variables]

# Setting seed and generating train-test split
set.seed(373)
sample <- createDataPartition(sentiment$label, p=0.82, list=FALSE)
training <- sentiment[sample, ]
testing <- sentiment[-sample, ]
nrow(training)
nrow(testing)

model_formula <- as.formula(paste("label ~", paste(selected_predictor_names, collapse = "+")))

# SVM
# svm.fit <- svm(model_formula, data = training, type = "C-classification", kernel = "linear")
# pred_train_svm <- predict(svm.fit, training)
# pred_test_svm <- predict(svm.fit, testing)
svm.fit <- svm(model_formula, data = training)
pred_train_svm_prob <- predict(svm.fit, training)
pred_test_svm_prob <- predict(svm.fit, testing)
pred_train_svm <- ifelse(pred_train_svm_prob >= 0.5, "1", "0")
pred_test_svm <- ifelse(pred_test_svm_prob >= 0.5, "1", "0")
confmat_train_svm <- table(Predicted = pred_train_svm, Actual = training$label)
training_accuracy_svm <- (confmat_train_svm[1, 1] + confmat_train_svm[2, 2]) / sum(confmat_train_svm) * 100
confmat_test_svm <- table(Predicted = pred_test_svm, Actual = testing$label)
testing_accuracy_svm <- (confmat_test_svm[1, 1] + confmat_test_svm[2, 2]) / sum(confmat_test_svm) * 100
testing_accuracy_svm

# Logistic Regression
logreg.fit <- glm(model_formula, data = training, family = binomial(link = "logit"))
pred_train_logreg_prob <- predict(logreg.fit, training, type = "response")
pred_test_logreg_prob <- predict(logreg.fit, testing, type = "response")
pred_train_logreg <- ifelse(pred_train_logreg_prob >= 0.5, "1", "0")
pred_test_logreg <- ifelse(pred_test_logreg_prob >= 0.5, "1", "0")
confmat_train_logreg <- table(Predicted = pred_train_logreg, Actual = training$label)
training_accuracy_logreg <- (confmat_train_logreg[1, 1] + confmat_train_logreg[2, 2]) / sum(confmat_train_logreg) * 100
confmat_test_logreg <- table(Predicted = pred_test_logreg, Actual = testing$label)
testing_accuracy_logreg <- (confmat_test_logreg[1, 1] + confmat_test_logreg[2, 2]) / sum(confmat_test_logreg) * 100
testing_accuracy_logreg

# probit
probit.fit <- glm(model_formula, data = training, family = binomial(link = "probit"))
pred_train_probit_prob <- predict(probit.fit, training, type = "response")
pred_test_probit_prob <- predict(probit.fit, testing, type = "response")
pred_train_probit <- ifelse(pred_train_probit_prob >= 0.5, "1", "0")
pred_test_probit <- ifelse(pred_test_probit_prob >= 0.5, "1", "0")
confmat_train_probit <- table(Predicted = pred_train_probit, Actual = training$label)
training_accuracy_probit <- (confmat_train_probit[1, 1] + confmat_train_probit[2, 2]) / sum(confmat_train_probit) * 100
confmat_test_probit <- table(Predicted = pred_test_probit, Actual = testing$label)
testing_accuracy_probit <- (confmat_test_probit[1, 1] + confmat_test_probit[2, 2]) / sum(confmat_test_probit) * 100
testing_accuracy_probit

# Naive Bayes
nb.fit <- naiveBayes(model_formula, data = training)
pred_train_nb_prob <- predict(nb.fit, training, type = "raw")
pred_test_nb_prob <- predict(nb.fit, testing, type = "raw")
pred_train_nb_prob <- pred_train_nb_prob[, "1"]
pred_test_nb_prob <- pred_test_nb_prob[, "1"]
pred_train_nb <- ifelse(pred_train_nb_prob >= 0.5, "1", "0")
pred_test_nb <- ifelse(pred_test_nb_prob >= 0.5, "1", "0")
confmat_train_nb <- table(Predicted = pred_train_nb, Actual = training$label)
training_accuracy_nb <- (confmat_train_nb[1, 1] + confmat_train_nb[2, 2]) / sum(confmat_train_nb) * 100
confmat_test_nb <- table(Predicted = pred_test_nb, Actual = testing$label)
testing_accuracy_nb <- (confmat_test_nb[1, 1] + confmat_test_nb[2, 2]) / sum(confmat_test_nb) * 100
testing_accuracy_nb

# K-Nearest Neighbors
knn.fit <- knn(train = training[selected_predictor_names], 
               test = testing[selected_predictor_names], 
               cl = training$label, 
               k = 3,
               prob = TRUE)
pred_train_knn_raw <- knn(train = training[selected_predictor_names], 
                          test = training[selected_predictor_names], 
                          cl = training$label, 
                          k = 3,
                          prob = TRUE)

pred_test_knn_raw <- knn.fit
pred_train_knn <- pred_train_knn_raw[1:nrow(training)]
pred_test_knn <- pred_test_knn_raw[1:nrow(testing)]
pred_train_knn_prob <- attr(pred_train_knn_raw, "prob")
pred_test_knn_prob <- attr(pred_test_knn_raw, "prob")
pred_train_knn_prob[pred_train_knn != "1"] = 1 - pred_train_knn_prob[pred_train_knn != 1]
pred_test_knn_prob[pred_test_knn != "1"] = 1 - pred_test_knn_prob[pred_test_knn != 1]
confmat_train_knn <- table(Predicted = pred_train_knn, Actual = training$label)
training_accuracy_knn <- (confmat_train_knn[1, 1] + confmat_train_knn[2, 2]) / sum(confmat_train_knn) * 100
training_accuracy_knn
confmat_test_knn <- table(Predicted = pred_test_knn, Actual = testing$label)
testing_accuracy_knn <- (confmat_test_knn[1, 1] + confmat_test_knn[2, 2]) / sum(confmat_test_knn) * 100
testing_accuracy_knn

# Decision Tree
dt_fit <- rpart(model_formula, data = training, method = "class")
pred_train_dt_prob <- predict(dt_fit, newdata = training, type = "prob")
pred_test_dt_prob <- predict(dt_fit, newdata = testing, type = "prob")
pred_train_dt_prob <- pred_train_dt_prob[, "1"]
pred_test_dt_prob <- pred_test_dt_prob[, "1"]
pred_train_dt <- ifelse(pred_train_dt_prob >= 0.5, "1", "0")
pred_test_dt <- ifelse(pred_test_dt_prob >= 0.5, "1", "0")
confmat_train_dt <- table(Predicted = pred_train_dt, Actual = training$label)
training_accuracy_dt <- sum(diag(confmat_train_dt)) / sum(confmat_train_dt) * 100
confmat_test_dt <- table(Predicted = pred_test_dt, Actual = testing$label)
testing_accuracy_dt <- sum(diag(confmat_test_dt)) / sum(confmat_test_dt) * 100
testing_accuracy_dt

# Ensemble using hard voting
pred_train_ensemble_hard <- (
                        as.numeric(pred_train_svm) +
                        as.numeric(pred_train_logreg) +
                        as.numeric(pred_train_nb) +
                        as.numeric(pred_train_knn)-1 + 
                        as.numeric(pred_train_dt) +
                        as.numeric(pred_train_probit)
                        )
pred_train_ensemble_hard <- ifelse(pred_train_ensemble_hard >= 5, "1", "0")
pred_test_ensemble_hard <- (
                       as.numeric(pred_test_svm) +
                       as.numeric(pred_test_logreg) +
                       as.numeric(pred_test_nb) +
                       as.numeric(pred_test_knn)-1 + 
                       as.numeric(pred_test_dt) +
                       as.numeric(pred_test_probit)
                       )
pred_test_ensemble_hard <- ifelse(pred_test_ensemble_hard >= 5, "1", "0")
confmat_train_ensemble_hard <- table(Predicted = pred_train_ensemble_hard, Actual = training$label)
training_accuracy_ensemble_hard <- (confmat_train_ensemble_hard[1, 1] + confmat_train_ensemble_hard[2, 2]) / sum(confmat_train_ensemble_hard) * 100
confmat_test_ensemble_hard <- table(Predicted = pred_test_ensemble_hard, Actual = testing$label)
testing_accuracy_ensemble_hard <- (confmat_test_ensemble_hard[1, 1] + confmat_test_ensemble_hard[2, 2]) / sum(confmat_test_ensemble_hard) * 100
testing_accuracy_ensemble_hard

# Ensemble using soft voting (probability)
pred_train_ensemble_soft <- (
    (
    as.numeric(pred_train_svm_prob) +
    as.numeric(pred_train_logreg_prob) +
    as.numeric(pred_train_nb_prob) +
    as.numeric(pred_train_knn_prob) +
    as.numeric(pred_train_dt_prob) +
    as.numeric(pred_train_probit_prob)
    ) / 6
)
pred_train_ensemble_soft <- ifelse(pred_train_ensemble_soft >= 0.52, "1", "0")
pred_test_ensemble_soft <- (
    (
    as.numeric(pred_test_svm_prob) +
    as.numeric(pred_test_logreg_prob) +
    as.numeric(pred_test_nb_prob) +
    as.numeric(pred_test_knn_prob) +
    as.numeric(pred_test_dt_prob) +
    as.numeric(pred_test_probit_prob)
    ) / 6
)
pred_test_ensemble_soft <- ifelse(pred_test_ensemble_soft >= 0.52, "1", "0")
confmat_train_ensemble_soft <- table(Predicted = pred_train_ensemble_soft, Actual = training$label)
training_accuracy_ensemble_soft <- (confmat_train_ensemble_soft[1, 1] + confmat_train_ensemble_soft[2, 2]) / sum(confmat_train_ensemble_soft) * 100
confmat_test_ensemble_soft <- table(Predicted = pred_test_ensemble_soft, Actual = testing$label)
testing_accuracy_ensemble_soft <- (confmat_test_ensemble_soft[1, 1] + confmat_test_ensemble_soft[2, 2]) / sum(confmat_test_ensemble_soft) * 100
testing_accuracy_ensemble_soft



# Ensemble using stacking method
meta_train <- data.frame(
  SVM = pred_train_svm_prob,
  LogReg = pred_train_logreg_prob,
  Probit = pred_train_probit_prob,
  NaiveBayes = pred_train_nb_prob,
  kNN = pred_train_knn_prob,
  DecisionTree = pred_train_dt_prob,
  label = training$label
)
meta_test <- data.frame(
  SVM = pred_test_svm_prob,
  LogReg = pred_test_logreg_prob,
  Probit = pred_test_probit_prob,
  NaiveBayes = pred_test_nb_prob,
  kNN = pred_test_knn_prob,
  DecisionTree = pred_test_dt_prob
)

meta_model <- glm(label ~ ., data = meta_train, family = binomial(link = "logit"))
pred_train_ensemble_stacking <- predict(meta_model, newdata = meta_train, type = "response")
pred_train_ensemble_stacking <- ifelse(pred_train_ensemble_stacking >= 0.52, "1", "0")
pred_test_ensemble_stacking <- predict(meta_model, newdata = meta_test, type = "response")
pred_test_ensemble_stacking <- ifelse(pred_test_ensemble_stacking >= 0.52, "1", "0")

confmat_train_ensemble_stacking <- table(Predicted = pred_train_ensemble_stacking, Actual = training$label)
training_accuracy_ensemble_stacking <- (confmat_train_ensemble_stacking[1, 1] + confmat_train_ensemble_stacking[2, 2]) / sum(confmat_train_ensemble_stacking) * 100
confmat_test_ensemble_stacking <- table(Predicted = pred_test_ensemble_stacking, Actual = testing$label)
testing_accuracy_ensemble_stacking <- (confmat_test_ensemble_stacking[1, 1] + confmat_test_ensemble_stacking[2, 2]) / sum(confmat_test_ensemble_stacking) * 100
training_accuracy_ensemble_stacking
testing_accuracy_ensemble_stacking

# Print out the accuracy of models
c(training_accuracy_svm, testing_accuracy_svm)
c(training_accuracy_logreg, testing_accuracy_logreg)
c(training_accuracy_probit, testing_accuracy_probit)
c(training_accuracy_nb, testing_accuracy_nb)
c(training_accuracy_knn, testing_accuracy_knn)
c(training_accuracy_dt, testing_accuracy_dt)
c(training_accuracy_ensemble_hard, testing_accuracy_ensemble_hard)
c(training_accuracy_ensemble_soft, testing_accuracy_ensemble_soft)
c(training_accuracy_ensemble_stacking, testing_accuracy_ensemble_stacking)
