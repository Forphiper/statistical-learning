#install.packages("car")
#install.packages("lmtest")
#install.packages("AER")
#install.packages("mfx")
#install.packages("caret")

#1.
## load default data from ISLR
library(ISLR)
data(Default)

# 2a.
dim(Default)

# 2b.
## check out class of variables
length(names(Default))
num_factor = 0
num_numerical = 0
for (col in names(Default)) {
    col_class <- class(Default[[col]])
    cat(paste0("Column '", col, "' has class '", col_class, "'\n"))
    if (col_class == 'factor') {
        num_factor <- num_factor + 1
    }
    else if (col_class == 'numerical') {
        num_numerical <- num_numerical + 1
    }
}
num_factor
num_numerical

# 2c.
## convert data of factor class to numerical
for (col in names(Default)) {
    col_class <- class(Default[[col]])
    if(col_class == 'factor') {
        Default[[col]] <- ifelse(Default[[col]] == 'Yes', 1, 0)
    }   
}
head(Default)

# 2d.
## fit a logistic regression model
logit.fit <- glm(default ~ student + balance + income + student:income, 
                 data = Default, family=binomial(link="logit"))
summary(logit.fit)

# 2e.
## transform the coefficients of your result to “odds-ratio” form
logit.coef <- exp(logit.fit$coefficients)
logit.coef

# 3.
## load the HMDA data and inspect the data
library(AER)
data(HMDA)

dim(HMDA)
names(HMDA)

num_factor = 0
num_numeric = 0
for (col in names(HMDA)) {
    col_class <- class(HMDA[[col]])
    cat(paste0("Column '", col, "' has class '", col_class, "'\n"))
    if (col_class == 'factor') {
        num_factor <- num_factor + 1
    }
    else if (col_class == 'numeric') {
        num_numeric <- num_numeric + 1
    }
}
num_factor
num_numeric

# 4a.
## convert the deny , selfemp, and insurance variables into binary numeric form
HMDA$deny <- ifelse(HMDA$deny == 'yes', 1, 0)
HMDA$selfemp <- ifelse(HMDA$selfemp == 'yes', 1, 0)
HMDA$insurance <- ifelse(HMDA$insurance == 'yes', 1, 0)

## probit models
model1 <- glm(deny ~ lvrat + selfemp + insurance, data = HMDA, family = binomial(link = "probit"))
model2 <- glm(deny ~ lvrat + I(lvrat^2) + selfemp + insurance, data = HMDA, family = binomial(link = "probit"))

summary(model1)
summary(model2)

# 4b.
## implement a “likelihood-ratio test” on the two models
lrtest(model1, model2)

# 4c.
## redo model2 with mfx
library(mfx)

model2_mfx <- probitmfx(deny ~ lvrat + I(lvrat^2) + selfemp + insurance, data = HMDA)
model2_mfx$mfxest

# 5a.
library(caret)
set.seed(6666)

## 75-25 train-test split
train_index <- createDataPartition(HMDA$lvrat, p = 0.75, list = FALSE)
training <- HMDA[train_index, ]
testing <- HMDA[-train_index, ]

# 5b.
## logistic model
training.fit <- glm(deny ~ lvrat + I(lvrat^2) + selfemp + insurance, data = training, family = binomial(link = "logit"))
pred_prob <- predict(training.fit, newdata = testing, type = "response")

# 5c.
## combine the true value of the dependent variable in the testing data with the predicted probability
results <- cbind(testing$deny, pred_prob)
head(results, 20)

