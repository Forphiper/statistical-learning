# install.packages("ISLR")
# install.packages("coefplot")
# install.packages("ggplot2")
# install.packages("gridExtra")
# install.packages("lmridge")

# 1. load packages and data
library(ISLR)
library(coefplot)
library(ggplot2)
library(gridExtra)
library(grid)
library(lmridge)
data(Carseats)

# 2a. data dimension
dim(Carseats)

# 2b. variables
names(Carseats)

# 2c. summary
summary(Carseats)

# 3a. model 1
lm.fit.a <- lm(Sales ~ Price + Advertising, data = Carseats)
summary(lm.fit.a)

# 3b. model 2
lm.fit.b <- lm(Sales ~ Price + Advertising + Education, data = Carseats[1:200, ])
summary(lm.fit.b)

# 3c. visualize
coefplot::coefplot.lm(lm.fit.a, predictors = c("Price", "Advertising"), parm = -1) + theme_bw()
coefplot::coefplot.lm(lm.fit.b, predictors = c("Price", "Advertising", "Education"), parm = -1) + theme_bw()

# 4. model 3
lm.fit.c <- lm(Sales ~ Income + Age + Education + Price, data = Carseats)
summary(lm.fit.c)
names(lm.fit.c)
mse <- mean(residuals(lm.fit.c)^2)
mse
rmse <- sqrt(mse)
rmse
rss <- sum(residuals(lm.fit.c)^2)
rss
rse <- sqrt(sum(residuals(lm.fit.c)^2) / lm.fit.c$df.residual)
rse
tss <- rss / (1 - summary(lm.fit.c)$r.squared)
tss
anova(lm.fit.c)

# 5b. two linear models
attach(Carseats)
lm.fit.a <- lm(Sales ~ Income + Age)
lm.fit.b <- lm(Sales ~ Income + Age + Education + Price)

# bias (aka., the MSE)
calc_bias = function(predicted, actual) {
   mean(sum((predicted - actual)^2))
}

# fitted values of your lm output are the predicted values
bias1 <- calc_bias(lm.fit.a$fitted.values, Sales)
bias2 <- calc_bias(lm.fit.b$fitted.values, Sales)

# variance (the sum of variance-covariance matrix)
var1 <- sum(vcov(lm.fit.a))
var2 <- sum(vcov(lm.fit.b))

# rMSE (the square root of MSE)
calc_rmse = function(actual, predicted) {
  sqrt(mean((actual - predicted) ^ 2))
}
rmse1 <- calc_rmse(Sales, lm.fit.a$fitted.values)
rmse2 <- calc_rmse(Sales, lm.fit.b$fitted.values)

table <- data.frame("Model" = c("Model 1", "Model 2"), "Bias" = c(bias1, bias2), "Variance" = c(var1, var2), "RMSE" = c(rmse1, rmse2))
grid.table(table)


#mod <- lmridge(Sales~Income+Age, Carseats, K = seq(-0.5, 0.3, 0.002))
#par(mar=c(2,2,1,1))  # pop-up browser layout mar=c(bottom, left, top, right)
#bias.plot(mod, abline = TRUE)
#legend(0.1, 12000, legend=c("Variance", "Bias^2", "MSE"),
#       col=c("black", "red", "green"), lty=c(1, 4, 2), cex=0.8)
