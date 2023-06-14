# 1a.
library(dplyr)
library(ggplot2)
library(tidyr)
library(caret)
library(skimr)
library(psych)
library(e1071)

## read pokemon data
pokemon = read.csv("https://www.dropbox.com/s/znbta9u9tub2ox9/pokemon.csv?dl=1")
dim(pokemon)
names(pokemon)
head(pokemon)

## there's one pokemon that has two capture_rate
poke <- pokemon
poke$capture_rate <- as.numeric(pokemon$capture_rate)
which(is.na(poke$capture_rate), arr.ind=TRUE)
pokemon[774, c("japanese_name", "capture_rate")]
pokemon[774, c("capture_rate")] <- 30
pokemon[774, c("japanese_name", "capture_rate")]

# 1b.
## subset the data
classify_legendary = select(pokemon, is_legendary, hp, weight_kg, height_m, 
                            sp_attack, sp_defense, type1, capture_rate)
colnames(classify_legendary)[7] <- "type"
head(classify_legendary)
length(unique(classify_legendary$type))

# 1c.
## replace NA values with 0
classify_legendary$weight_kg[is.na(classify_legendary$weight_kg)] <- 0
classify_legendary$height_m[is.na(classify_legendary$height_m)] <- 0
colSums(is.na(classify_legendary)) # check if NA values still exist

class(classify_legendary$capture_rate)
classify_legendary$capture_rate <- as.numeric(classify_legendary$capture_rate)
class(classify_legendary$capture_rate)

class(classify_legendary$is_legendary)
classify_legendary$is_legendary <- as.factor(classify_legendary$is_legendary)
class(classify_legendary$is_legendary)

# 1d.
## one-hot encoding
## dummyVars() creates a full set of dummy variables for the formula provided
dummies_model <- dummyVars(is_legendary ~ ., data = classify_legendary)
## use predict() to map this function to the dataframe you want to convert
data_mat <- predict(dummies_model, newdata = classify_legendary) # one-hot-encode all categorical variables
classify_legendary_ohe <- data.frame(data_mat) # convert data_mat into dataframe class
classify_legendary_ohe$is_legendary <- classify_legendary$is_legendary
head(classify_legendary_ohe)

length(names(classify_legendary))
length(names(classify_legendary_ohe))
names(classify_legendary_ohe)

# 1e.(i)
## 75-25 train-test split on Y before testing the model
set.seed(123)
train_row_numbers_ohe <- createDataPartition(classify_legendary_ohe$is_legendary, p=0.75, list=FALSE)
train_classify_legendary_ohe <- classify_legendary_ohe[train_row_numbers_ohe, ]
test_classify_legendary_ohe <- classify_legendary_ohe[-train_row_numbers_ohe, ]

# 1e.(ii)
## SVM
svm_ohe.fit <- svm(is_legendary~., 
                   data = train_classify_legendary_ohe)

# 1e.(iii)
predict_test_svm_ohe <- predict(svm_ohe.fit, test_classify_legendary_ohe)

# 1e.(iv)
## generate confusion matrix and show accuracy
confmat_test_svm_ohe <- table(Predicted = predict_test_svm_ohe, Actual = test_classify_legendary_ohe$is_legendary)
(confmat_test_svm_ohe[1, 1] + confmat_test_svm_ohe[2, 2]) / sum(confmat_test_svm_ohe) * 100

## do the same fitting to the no-ohe dataframe
set.seed(123)
train_row_numbers <- createDataPartition(classify_legendary$is_legendary, p=0.75, list=FALSE)
train_classify_legendary <- classify_legendary[train_row_numbers, ]
test_classify_legendary <- classify_legendary[-train_row_numbers, ]
svm.fit <- svm(is_legendary~.,
                   data = train_classify_legendary)
predict_test_svm <- predict(svm.fit, test_classify_legendary)
confmat_test_svm <- table(Predicted = predict_test_svm, Actual = test_classify_legendary$is_legendary)
(confmat_test_svm[1, 1] + confmat_test_svm[2, 2]) / sum(confmat_test_svm) * 100

# 1f.
## repeat 1a-1b
pokemon = read.csv("https://www.dropbox.com/s/znbta9u9tub2ox9/pokemon.csv?dl=1")
poke <- pokemon
poke$capture_rate <- as.numeric(pokemon$capture_rate)
which(is.na(poke$capture_rate), arr.ind=TRUE)
pokemon[774, c("japanese_name", "capture_rate")]
pokemon[774, c("capture_rate")] <- 30
pokemon[774, c("japanese_name", "capture_rate")]
classify_legendary = select(pokemon, is_legendary, hp, weight_kg, height_m, 
                            sp_attack, sp_defense, type1, capture_rate)
colnames(classify_legendary)[7] <- "type"
head(classify_legendary)

classify_legendary$capture_rate <- as.numeric(classify_legendary$capture_rate)
classify_legendary$is_legendary <- as.factor(classify_legendary$is_legendary)

class(classify_legendary$capture_rate)
class(classify_legendary$is_legendary)
colSums(is.na(classify_legendary))
head(classify_legendary)

## use KNN to impute the missing values
library(RANN)
pre_process_missing_data <- preProcess(as.data.frame(classify_legendary), method="knnImpute")
## in order for a data to be processed via preProcess(), the first argument needs to be of data.frame class. I found that preProcess() is mostly compatible with tibble and other classes of inputs, but it does require programmers to convert his/her data into data.frame class when applying knnImpute method. Some colleagues have reported that they encounter no such problems though.
classify_legendary <- predict(pre_process_missing_data, newdata = classify_legendary)
colSums(is.na(classify_legendary))
head(classify_legendary)

## one-hot encoding
## dummyVars() creates a full set of dummy variables for the formula provided
dummies_model <- dummyVars(is_legendary ~ ., data = classify_legendary)
## use predict() to map this function to the dataframe you want to convert
data_mat <- predict(dummies_model, newdata = classify_legendary) # one-hot-encode all categorical variables
classify_legendary_ohe <- data.frame(data_mat) # convert data_mat into dataframe class
classify_legendary_ohe$is_legendary <- classify_legendary$is_legendary
head(classify_legendary_ohe)

length(names(classify_legendary))
length(names(classify_legendary_ohe))
names(classify_legendary_ohe)

## fit the model with ohe data
set.seed(123)
train_row_numbers_ohe <- createDataPartition(classify_legendary_ohe$is_legendary, p=0.75, list=FALSE)
train_classify_legendary_ohe <- classify_legendary_ohe[train_row_numbers_ohe, ]
test_classify_legendary_ohe <- classify_legendary_ohe[-train_row_numbers_ohe, ]
svm_ohe.fit <- svm(is_legendary~., 
                   data = train_classify_legendary_ohe)
predict_test_svm_ohe <- predict(svm_ohe.fit, test_classify_legendary_ohe)
confmat_test_svm_ohe <- table(Predicted = predict_test_svm_ohe, Actual = test_classify_legendary_ohe$is_legendary)
(confmat_test_svm_ohe[1, 1] + confmat_test_svm_ohe[2, 2]) / sum(confmat_test_svm_ohe) * 100

## fit the model without ohe
set.seed(123)
train_row_numbers <- createDataPartition(classify_legendary$is_legendary, p=0.75, list=FALSE)
train_classify_legendary <- classify_legendary[train_row_numbers, ]
test_classify_legendary <- classify_legendary[-train_row_numbers, ]
svm.fit <- svm(is_legendary~.,
               data = train_classify_legendary)
predict_test_svm <- predict(svm.fit, test_classify_legendary)
confmat_test_svm <- table(Predicted = predict_test_svm, Actual = test_classify_legendary$is_legendary)
(confmat_test_svm[1, 1] + confmat_test_svm[2, 2]) / sum(confmat_test_svm) * 100

# 2a.
library(e1071)
library(caret)
library(gbm)
library(pROC)
library(plotROC)
library(ISLR)
library(glmnet)
library(dplyr)

data(Default)
dim(Default)
head(Default)

# 2b.(i)
## 75-25 train-test split
set.seed(123)
train_row_numbers <- createDataPartition(Default$default, p=0.75, list=FALSE)
training <- Default[train_row_numbers, ]
testing <- Default[-train_row_numbers, ]
nrow(training)
nrow(testing)

# training$student <- ifelse(training$student == "No", 0, 1)
# testing$student <- ifelse(testing$student == "No", 0, 1)

# 2b.(ii)
## logit
logit.fit <- glm(default~., data = training, family = binomial(link = "logit"))

## probit
probit.fit <- glm(default~., data = training, family = binomial(link = "probit"))

## SVM
svm.fit <- svm(default~., data = training)

## GBM
training_gbm = training
testing_gbm = testing
training_gbm$default <- ifelse(training_gbm$default == "No", 0, 1)
testing_gbm$default <- ifelse(testing_gbm$default == "No", 0, 1)

hyper_grid1 <- expand.grid(
  shrinkage = c(.01, .1, .2),
  interaction.depth = c(1, 3, 5),
  n.minobsinnode = c(5, 10, 15),
  bag.fraction = c(.5, .75, 1),
  optimal_trees = 0, # you will fill in values from loop
  min_RMSE = 0) # you will fill in values from loop

## Iterate through hyperparameters
for(i in 1:nrow(hyper_grid1)) {
  
  # train model
  gbm.tune <- gbm(
    default ~ .,
    distribution = "bernoulli",
    data = training_gbm,
    cv.folds = 5,
    n.trees = 500, # number of trees
    interaction.depth = hyper_grid1$interaction.depth[i], # tree depth
    shrinkage = hyper_grid1$shrinkage[i], # learning rate (the rate at which the gradient descends)
    n.minobsinnode = hyper_grid1$n.minobsinnode[i], # number of observation(s) in each terminal node
    bag.fraction = hyper_grid1$bag.fraction[i], # fraction of sample for sub-sampling
    train.fraction = .75,
    n.cores = NULL, # will use all cores by default
    verbose = FALSE  # do not print out progress
  )
  
  # locate minimum training error from the n-th tree, add it to grid
  hyper_grid1$optimal_trees[i] <- which.min(gbm.tune$valid.error)
  hyper_grid1$min_RMSE[i] <- sqrt(min(gbm.tune$valid.error))
}

## display 10 best performing models in descending order
optimal_para <- 
  hyper_grid1 %>% 
  dplyr::arrange(min_RMSE) %>%
  head(10)
optimal_para

## use the best hyperparameters to fit the model
gbm.fit <- gbm(
    default ~ .,
    data = training_gbm,
    distribution = "bernoulli",
    cv.folds = 5,
    interaction.depth = optimal_para$interaction.depth[1],
    shrinkage = optimal_para$shrinkage[1],
    n.trees = optimal_para$optimal_trees[1],
    n.minobsinnode = optimal_para$n.minobsinnode[1],
    bag.fraction = optimal_para$bag.fraction[1],
    n.cores = NULL,
    verbose = FALSE)

# 2b.(iii)
## predict with the four models
predict_test_logit <- predict(logit.fit, newdata = testing, type = "response")
predict_test_probit <- predict(probit.fit, newdata = testing, type = "response")
predict_test_svm <- predict(svm.fit, newdata = testing)
predict_test_gbm <- predict(gbm.fit, newdata = testing_gbm, type = "response")

predict_logit_class <- as.numeric(predict_test_logit > 0.5)
predict_probit_class <- as.numeric(predict_test_probit > 0.5)
predict_svm_class <- as.numeric(predict_test_svm)
predict_gbm_class <- as.numeric(predict_test_gbm > 0.5)

# 2b.(iv)
## generate confusion table
confmat_test_logit <- table(Predicted = predict_logit_class, Actual = testing$default)
confmat_test_probit <- table(Predicted = predict_probit_class, Actual = testing$default)
confmat_test_svm <- table(Predicted = predict_svm_class, Actual = testing$default)
confmat_test_gbm <- table(Predicted = predict_gbm_class, Actual = testing_gbm$default)

## show precision scores
confmat_test_logit[2, 2] / (confmat_test_logit[2, 1] + confmat_test_logit[2, 2])
confmat_test_probit[2, 2] / (confmat_test_probit[2, 1] + confmat_test_probit[2, 2])
confmat_test_svm[2, 2] / (confmat_test_svm[2, 1] + confmat_test_svm[2, 2])
confmat_test_gbm[2, 2] / (confmat_test_gbm[2, 1] + confmat_test_gbm[2, 2])

# 2c.
## convert default back to factor
class(training_gbm$default)
training_gbm$default <- as.factor(training_gbm$default)
class(training_gbm$default)

class(testing_gbm$default)
testing_gbm$default <- as.factor(testing_gbm$default)
class(testing_gbm$default)

## generate ROC objects
ROC_logit <- roc(testing$default, predict_logit_class)
ROC_probit <- roc(testing$default, predict_probit_class)
ROC_svm <- roc(testing$default, predict_svm_class)
ROC_gbm <- roc(testing_gbm$default, predict_gbm_class)

# 2d.
## calculate AUC with ROC objects
auc(ROC_logit)
auc(ROC_probit)
auc(ROC_svm)
auc(ROC_gbm)

# 2e.
## visualize
par(mar=c(1,1,1,1))

plot(ROC_logit, xlim = c(1.1, -0.1))
lines(ROC_probit, col="blue")
lines(ROC_svm, col="red")
lines(ROC_gbm, col="green")
legend(0.2, 0.5,
       legend = c("model   AUC", "logit:   0.6477", "probit: 0.6184", "SVM:  0.601", "GBM:  0.6537"),
       col = c("white", "black", "blue", "red", "green"),
       lty = c(1, 1, 2, 3, 4),
       pch = c(NA, NA, NA, NA, NA),
       cex = 0.7)

# 3a.
library(dplyr)
library(rsample)
library(boot)
library(purrr)
library(tictoc)

data(Default)
dim(Default)
head(Default)

## create a function that computes k-fold MSE based on specified polynomial degree
kfcv_error <- function(x) {
  set.seed(123)
  glm.fit <- glm(default ~ student + balance + poly(income, x), data = Default, family = "binomial")
  cv.glm(Default, glm.fit, K = 10)$delta[1]  # I set K = 10
}

## compute k-fold MSE for polynomial degrees 1-10 and time it
tic()
1:10 %>% map_dbl(kfcv_error)
toc()

# 3b.
## income of degree 1 is the best
poly.result <- function(data, index) {
  set.seed(123)
  glm.fit <- glm(default ~ student + balance + poly(income, 1), 
                 data = Default, family = "binomial", subset = index)
  coef(glm.fit)
}

## now we turn to boot()
set.seed(123)
boot(Default, poly.result, 100)

