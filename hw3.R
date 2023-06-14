# 1a. 
#install.packages("ISLR")
#install.packages("tidyverse")
#install.packages("ggplot2")
#install.packages("dplyr")
#install.packages("caret")

library(ISLR)
library(tidyverse)
library(ggplot2)
library(dplyr)
library(caret)

data(Credit)
dim(Credit)
names(Credit)
head(Credit)

# 1b.
## Setting seed and generating train-test split
set.seed(123)
nrow(Credit)
sample <- createDataPartition(Credit$Rating, p=0.75, list=FALSE)
training <- Credit[sample, ]
testing <- Credit[-sample, ]

nrow(training)
nrow(testing)

## Fitting linear regression models
poly.5 = lm(Rating ~ poly(Income, 5), data = training)
summary(poly.5)

# 1b. bonus
poly.5.raw = lm(Rating ~ poly(Income, 5, raw = TRUE), data = training)
summary(poly.5.raw)

# 1c.
## Fit linear regression
lm.trans <- lm(Rating ~ log(Income), data = training)
summary(lm.trans)

# 1d.
## Predicting on the testing data
poly5.pred <- poly.5 %>% predict(testing)
trans.pred <- lm.trans %>% predict(testing)

comparison <- data.frame(
  model = c("Poly degree 5", "log-transformed"),
  RMSE = c(RMSE(poly5.pred, testing$Rating), RMSE(trans.pred, testing$Rating)),
  R2 = c(R2(poly5.pred, testing$Rating), R2(trans.pred, testing$Rating))
)
comparison

# 1e.
## Plot the results
ggplot(training, aes(Income, Rating) ) +
  geom_point(alpha = 0.2) +         # using alpha to adjust transparency level
  labs(x = "Income", y = "Rating", title = "Predicted Rating as a function of Income") +
  stat_smooth(method = lm, formula = y ~ poly(x, 5), col = "red")  # enter your model specification in the formula argument

ggplot(training, aes(log(Income), Rating) ) +
  geom_point(alpha = 0.2) +         # using alpha to adjust transparency level
  labs(x = "Income", y = "Rating", title = "Predicted Rating as a function of Income") +
  stat_smooth(method = lm, formula = y ~ poly(x), col = "red")  # enter your model specification in the formula argument

# 2a.
## Load the data
data <- read.csv("https://www.dropbox.com/s/11rmse4ay9uu8vu/Social_Network_Ads.csv?dl=1")
dim(data)
head(data)

# 2b.
## Subset the data
social_ads <- data[, c("Age", "EstimatedSalary", "Purchased")]
head(social_ads)

# 2c.
## Convert purchased to factor
class(social_ads$Purchased)
social_ads$Purchased <- as.factor(social_ads$Purchased)
class(social_ads$Purchased)

# 2d.
## 75-25 train-test split
set.seed(123)
train_index <- createDataPartition(social_ads$Purchased, p=0.75, list=FALSE)
train_data <- social_ads[train_index, ]
test_data <- social_ads[-train_index, ]
nrow(train_data)
nrow(test_data)

# 2e. 
## Fit a SVM model
library(e1071)
svm.fit <- svm(Purchased ~ Age + EstimatedSalary, data = train_data, type = "C-classification", kernel = "linear")
summary(svm.fit)

# 2f.
## Predict on testing data
pred_train_svm <- predict(svm.fit, train_data)
pred_test_svm <- predict(svm.fit, test_data)

# 2g.
## Generate confusion matrix and show prediction accuracy
confmat_train_svm <- table(Predicted = pred_train_svm, 
                          Actual = train_data$Purchased)
confmat_train_svm
print("training accuracy")
(confmat_train_svm[1, 1] + confmat_train_svm[2, 2]) / sum(confmat_train_svm) * 100

confmat_test_svm <- table(Predicted = pred_test_svm, 
                           Actual = test_data$Purchased)
confmat_test_svm
print("testing accuracy")
(confmat_test_svm[1, 1] + confmat_test_svm[2, 2]) / sum(confmat_test_svm) * 100

# 2h.
plot(svm.fit, train_data)

# 3a
#install.packages("ISLR")
#install.packages("glmnet")
#install.packages("plotmo")
library(ISLR)
library(glmnet)
library(plotmo)

data(Hitters)
dim(Hitters)
head(Hitters)

# 3b
## Remove missing values
Hitters <- na.omit(Hitters)
dim(Hitters)

## Fit a LASSO model
y <- Hitters$Salary
X <- model.matrix(Salary ~ .-1, data = Hitters)
LASSO.fit <- glmnet(X, y, alpha = 1)
print(LASSO.fit)

# 3c (i)
## Plot top 5 variables at optimal lambda value
plot_glmnet(LASSO.fit, xvar = "lambda", label = 5, 
            xlab = expression(paste("log(", lambda, ")")),   # use expression() to ask R to enable Greek letters typesetting
            ylab = expression(beta))
plot_glmnet(LASSO.fit, label = 5, xlab = expression(paste("log(", lambda, ")")), ylab = expression(beta))

## display the estimated coef
coef(LASSO.fit, s = cv.glmnet(X, y)$lambda.1se)

# 3c (ii)
## Extract coefficients for lambda = 1
coef(LASSO.fit, s = 1)






