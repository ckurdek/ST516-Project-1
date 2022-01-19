require(rpart)
require(randomForest)
require(caret)
require(dplyr)
require(gbm)
require(glmnet)
require(tree)
library(boot)
library(car)
library(leaps)

housingData <-  read.csv("D:\\NCSU\\Semester 1\\ST516\\Project 1\\housingData.csv")
str(housingData)
dim(housingData)


#NA values
sum(is.na(housingData)) / (nrow(housingData) *ncol(housingData))
colSums(sapply(housingData, is.na))

Missing_indices <- sapply(housingData,function(x) sum(is.na(x)))
Missing_Summary <- data.frame(index = names(housingData),Missing_Values=Missing_indices)
Missing_Summary[Missing_Summary$Missing_Values > 0,]

m <- mean(housingData$LotFrontage[-which(is.na(housingData$LotFrontage))])

housingData$LotFrontage[which(is.na(housingData$LotFrontage))] <- as.integer(m)

housingData <- housingData%>% filter(is.na(BsmtFinType1)==F)
housingData <- housingData%>% filter(is.na(GarageType)==F)

#No. of Integer Variables
sum(sapply(housingData[,1:26], typeof) == "integer")
hdata_num <- housingData[,sapply(housingData[,1:26], typeof) == "integer"]
num_names <- names(hdata_num)

#No. of Character Variables
sum(sapply(housingData[,1:26], typeof) == "character")
hdata_char <- housingData[,sapply(housingData[,1:26], typeof) == "character"]
char_names <- names(hdata_char)
char_names

#Splitting the dataset into train and test in the ratio 80:20
x <- model.matrix(SalePrice~.,housingData)[,-1]

y <- housingData$SalePrice
z <- data.frame(x)
z$SalePrice <- y

set.seed(1234)
smp_size <- floor(0.8 * nrow(z))
train_ind=sample(seq_len(nrow(z)), size = smp_size, rep=F)

train_data_lm <- z[train_ind,]
test_data_lm <- z[-train_ind,]

# Transformation of Response variable
train_data_lm$SalePrice <- log(train_data_lm$SalePrice)

#Linear model

set.seed(1234)
lm_fit <- lm(SalePrice~.,data=train_data_lm)
lm.sum <- summary(lm_fit)
lm.sum$coefficients

lm_predict <- predict(lm_fit,test_data_lm)
mse1<-mean((test_data_lm$SalePrice-exp(lm_predict))^2)
sqrt(mse1)

# Regularization of linear models
set.seed(1234)
x_train <- model.matrix(SalePrice~.,train_data_lm)[,-1]
y_train <- train_data_lm$SalePrice

x_test <-  model.matrix(SalePrice~.,test_data_lm)[,-1]
y_test <- test_data_lm$SalePrice

#Ridge regression
set.seed(1234)
cv.ridge = cv.glmnet (x_train ,y_train,type.measure = "mse",alpha =0,family = "gaussian")
plot(cv.ridge) 
coef(cv.ridge)
cv.ridge.predict = predict(cv.ridge, s=cv.ridge$lambda.min, newx = x_test)
mse2 <- mean((y_test-exp(cv.ridge.predict))^2)
sqrt(mse2)

SSE_r <- sum((y_test-exp(cv.ridge.predict))^2)
SST_r <- sum((y_test-mean(y_test))^2)
r_sq_r <- 1 - SSE_r/SST_r
r_sq_r

#Fit least squares full model and compare to Ridge
lmod <- lm(SalePrice~.,data=z)
fit.ridge <- predict(cv.ridge,s=cv.ridge$lambda.min,model.matrix(SalePrice~.,z)[,-1])

plot(lmod$fitted.values,z$SalePrice,pch=19,col="blue")
points(exp(fit.ridge),z$SalePrice,col="red",lwd=2)
abline(a=0,b=1)
legend("topleft", legend=c("OLS", "Ridge"), pch=c(19, 1), col=c("blue", "red"), bty="n" )
z$SalePrice

#Lasso regression
set.seed(1234)
cv.lasso = cv.glmnet(x_train, y_train, type.measure = "mse", alpha = 1, family = "gaussian")
plot(cv.lasso)
coef(cv.lasso)
cv.lasso.predict = predict(cv.lasso, s=cv.lasso$lambda.min, newx= x_test)
mse3 <- mean((y_test - exp(cv.lasso.predict))^2)
sqrt(mse3)

SSE_l <- sum((y_test-exp(cv.lasso.predict))^2)
SST_l <- sum((y_test-mean(y_test))^2)
r_sq_lasso <- 1 - SSE_l/SST_l
r_sq_lasso

#Fit least squares full model and compare to lasso
lmod <- lm(SalePrice~.,data=z)
fit.lasso <- predict(cv.lasso,s=cv.lasso$lambda.min,model.matrix(SalePrice~.,z)[,-1])

plot(lmod$fitted.values,z$SalePrice,pch=19,col="blue")
points(exp(fit.ridge),z$SalePrice,col="red",lwd=2)
abline(a=0,b=1)
legend("topleft", legend=c("OLS", "Lasso"), pch=c(19, 1), col=c("blue", "red"), bty="n" )

# forward stepwise selection
regfit_fwd = regsubsets(SalePrice~., data = z, nvmax=87,method = "forward")
fwd.sum <- summary(regfit_fwd)

# can use criterion to select best model
p = 1:87
aic <- fwd.sum$bic+2*p-log(dim(z)[1])*p
which.max(fwd.sum$adjr2)
which.min(fwd.sum$bic)
which.min(aic)

# plot criteria to get visual confirmation
par(mfrow=c(2,2))
plot(p,aic,pch=19,type="b",main="AIC")
points(which.min(aic),aic[which.min(aic)],cex=1.5,col="red",lwd=2)
abline(v=c(1:87),lty=3)

plot(p,fwd.sum$bic,pch=19,type="b",main="BIC")
points(which.min(fwd.sum$bic),fwd.sum$bic[which.min(fwd.sum$bic)],
       cex=1.5,col="red",lwd=2)
abline(v=c(1:87),lty=3)

plot(p,fwd.sum$adjr2,pch=19,type="b",main="adj-R2")
points(which.max(fwd.sum$adjr2),fwd.sum$adjr2[which.max(fwd.sum$adjr2)],
       cex=1.5,col="red",lwd=2)
abline(v=c(1:87),lty=3)

#  fit best model for p=50 (AIC)
xvarm <- names(coef(regfit_fwd,id=50))[2:51]
housing.best <- z[,c("SalePrice",xvarm)]
lmod <- lm(SalePrice~.,data=housing.best)
par(mfrow=c(2,2))
plot(lmod)
summary(lmod)
vif(lmod)

#Splitting the dataset into train and test in the ratio 80:20
set.seed(1234)
smp_size <- floor(0.8 * nrow(housing.best))
train_ind=sample(seq_len(nrow(housing.best)), size = smp_size, rep=F)

train_data_fw <- housing.best[train_ind,]
test_data_fw <- housing.best[-train_ind,]

train_data_fw$SalePrice <- log(train_data_fw$SalePrice)

set.seed(1234)
lm_fit <- lm(SalePrice~.,data=train_data_fw)
summary(lm_fit)

lm_predict <- predict(lm_fit,test_data_fw)
msefw<-mean((test_data_fw$SalePrice-exp(lm_predict))^2)
sqrt(msefw)

# backward stepwise selection

regfit_bwd = regsubsets(SalePrice~., data = z, method = "backward", nvmax=87)
bkwd.sum <- summary(regfit_bwd)

aic <- bkwd.sum$bic+2*p-log(dim(z)[1])*p
which.max(bkwd.sum$adjr2)
which.min(bkwd.sum$bic)
which.min(aic)

# plot criteria to get visual confirmation

par(mfrow=c(2,2))
plot(p,aic,pch=19,type="b",main="AIC")
points(which.min(aic),aic[which.min(aic)],cex=1.5,col="red",lwd=2)
abline(v=c(1:87),lty=3)

plot(p,bkwd.sum$bic,pch=19,type="b",main="BIC")
points(which.min(bkwd.sum$bic),bkwd.sum$bic[which.min(bkwd.sum$bic)],
       cex=1.5,col="red",lwd=2)
abline(v=c(1:87),lty=3)

plot(p,bkwd.sum$adjr2,pch=19,type="b",main="adj-R2")
points(which.max(bkwd.sum$adjr2),bkwd.sum$adjr2[which.max(bkwd.sum$adjr2)],
       cex=1.5,col="red",lwd=2)
abline(v=c(1:87),lty=3)

#  fit best model for p=49 (AIC)

xvarm <- names(coef(regfit_bwd,id=49))[2:50]
housingbk.best <- z[,c("SalePrice",xvarm)]
lmod <- lm(SalePrice~.,data=housingbk.best)
par(mfrow=c(2,2))
plot(lmod)
summary(lmod)
vif(lmod)

dim(housingbk.best)

#Splitting the dataset into train and test in the ratio 80:20
set.seed(1234)
smp_size <- floor(0.8 * nrow(housingbk.best))
train_ind=sample(seq_len(nrow(housingbk.best)), size = smp_size, rep=F)

train_data_bk <- housingbk.best[train_ind,]
test_data_bk <- housingbk.best[-train_ind,]

train_data_bk$SalePrice <- log(train_data_bk$SalePrice)

set.seed(1234)
lm_fit <- lm(SalePrice~.,data=train_data_bk)
summary(lm_fit)

lm_predict <- predict(lm_fit,test_data_bk)
msebk<-mean((test_data_bk$SalePrice-exp(lm_predict))^2)
sqrt(msebk)

#Reduced data


train_data_plm <- train_data_lm[c('OverallQual','OverallCond','GrLivArea','YearBuilt','TotalBsmtSF','BsmtFinType1Unf','KitchenQualGd',
                                  'GarageCars','KitchenQualFa','Fireplaces','BldgTypeTwnhs','KitchenQualTA','LotArea','NeighborhoodStoneBr',
                                  'BldgTypeTwnhsE','BldgTypeDuplex','Exterior1stBrkFace','NeighborhoodMeadowV','Exterior1stMetalSd','NeighborhoodCrawfor',
                                  'NeighborhoodNridgHt','HouseStyleSLvl','FoundationPConc','Exterior1stVinylSd','LotFrontage','Exterior1stCemntBd',
                                  'Exterior1stWd.Sdng','Exterior1stPlywood','Exterior1stWdShing','YearRemodAdd','NeighborhoodSomerst','SalePrice')]
dim(train_data_plm)

test_data_plm <- test_data_lm[c('OverallQual','OverallCond','GrLivArea','YearBuilt','TotalBsmtSF','BsmtFinType1Unf','KitchenQualGd',
                                'GarageCars','KitchenQualFa','Fireplaces','BldgTypeTwnhs','KitchenQualTA','LotArea','NeighborhoodStoneBr',
                                'BldgTypeTwnhsE','BldgTypeDuplex','Exterior1stBrkFace','NeighborhoodMeadowV','Exterior1stMetalSd','NeighborhoodCrawfor',
                                'NeighborhoodNridgHt','HouseStyleSLvl','FoundationPConc','Exterior1stVinylSd','LotFrontage','Exterior1stCemntBd',
                                'Exterior1stWd.Sdng','Exterior1stPlywood','Exterior1stWdShing','YearRemodAdd','NeighborhoodSomerst','SalePrice')]


z_plm <- z[c('OverallQual','OverallCond','GrLivArea','YearBuilt','TotalBsmtSF','BsmtFinType1Unf','KitchenQualGd',
             'GarageCars','KitchenQualFa','Fireplaces','BldgTypeTwnhs','KitchenQualTA','LotArea','NeighborhoodStoneBr',
             'BldgTypeTwnhsE','BldgTypeDuplex','Exterior1stBrkFace','NeighborhoodMeadowV','Exterior1stMetalSd','NeighborhoodCrawfor',
             'NeighborhoodNridgHt','HouseStyleSLvl','FoundationPConc','Exterior1stVinylSd','LotFrontage','Exterior1stCemntBd',
             'Exterior1stWd.Sdng','Exterior1stPlywood','Exterior1stWdShing','YearRemodAdd','NeighborhoodSomerst','SalePrice')]
dim(z_plm)

#Reduced linear

train_data_plm_log <- train_data_plm

set.seed(1234)
lm_fit <- lm(SalePrice~.,data=train_data_plm_log)
lm.sum <- summary(lm_fit)
lm.sum

lm_predict <- predict(lm_fit,test_data_plm)
mse1red <-mean((test_data_plm$SalePrice-exp(lm_predict))^2)
sqrt(mse1red)
mse1red

#Fit least squares full model and compare to lasso
lmod <- lm(SalePrice~.,data=z)
fit.lmr <- predict(lm_fit,z_plm)

plot(lmod$fitted.values,z$SalePrice,pch=19,col="blue")
points(exp(fit.lmr),z_plm$SalePrice,col="red",lwd=2)
abline(a=0,b=1)
legend("topleft", legend=c("OLS", "Reduced LS"), pch=c(19, 1), col=c("blue", "red"), bty="n" )


# Reduced forward stepwise selection
regfit_fwd = regsubsets(SalePrice~., data = z_plm, nvmax=31,method = "forward")
fwd.sum <- summary(regfit_fwd)

# Criterion to select best model
p = 1:31
aic <- fwd.sum$bic+2*p-log(dim(z_plm)[1])*p
which.max(fwd.sum$adjr2)
which.min(fwd.sum$bic)
which.min(aic)

# plot criteria to get visual confirmation
par(mfrow=c(2,2))
plot(p,aic,pch=19,type="b",main="AIC")
points(which.min(aic),aic[which.min(aic)],cex=1.5,col="red",lwd=2)
abline(v=c(1:31),lty=3)

plot(p,fwd.sum$bic,pch=19,type="b",main="BIC")
points(which.min(fwd.sum$bic),fwd.sum$bic[which.min(fwd.sum$bic)],
       cex=1.5,col="red",lwd=2)
abline(v=c(1:31),lty=3)

plot(p,fwd.sum$adjr2,pch=19,type="b",main="adj-R2")
points(which.max(fwd.sum$adjr2),fwd.sum$adjr2[which.max(fwd.sum$adjr2)],
       cex=1.5,col="red",lwd=2)
abline(v=c(1:31),lty=3)

#  fit best model for p=24 (AIC)
xvarm <- names(coef(regfit_fwd,id=24))[2:25]
housing.best <- z_plm[,c("SalePrice",xvarm)]
lmod <- lm(SalePrice~.,data=housing.best)
par(mfrow=c(2,2))
plot(lmod)
summary(lmod)
vif(lmod)

#Splitting the dataset into train and test in the ratio 80:20
set.seed(1234)
smp_size <- floor(0.8 * nrow(housing.best))
train_ind=sample(seq_len(nrow(housing.best)), size = smp_size, rep=F)

train_data_fw <- housing.best[train_ind,]
test_data_fw <- housing.best[-train_ind,]

train_data_fw$SalePrice <- log(train_data_fw$SalePrice)

set.seed(1234)
lm_fit <- lm(SalePrice~.,data=train_data_fw)
summary(lm_fit)

lm_predict <- predict(lm_fit,test_data_fw)
msefw<-mean((test_data_fw$SalePrice-exp(lm_predict))^2)
sqrt(msefw)

# Reduced backward stepwise selection

regfit_bwd = regsubsets(SalePrice~., data = z_plm, method = "backward", nvmax=31)
bkwd.sum <- summary(regfit_bwd)

aic <- bkwd.sum$bic+2*p-log(dim(z_plm)[1])*p
which.max(bkwd.sum$adjr2)
which.min(bkwd.sum$bic)
which.min(aic)

# plot criteria to get visual confirmation

par(mfrow=c(2,2))
plot(p,aic,pch=19,type="b",main="AIC")
points(which.min(aic),aic[which.min(aic)],cex=1.5,col="red",lwd=2)
abline(v=c(1:31),lty=3)

plot(p,bkwd.sum$bic,pch=19,type="b",main="BIC")
points(which.min(bkwd.sum$bic),bkwd.sum$bic[which.min(bkwd.sum$bic)],
       cex=1.5,col="red",lwd=2)
abline(v=c(1:31),lty=3)

plot(p,bkwd.sum$adjr2,pch=19,type="b",main="adj-R2")
points(which.max(bkwd.sum$adjr2),bkwd.sum$adjr2[which.max(bkwd.sum$adjr2)],
       cex=1.5,col="red",lwd=2)
abline(v=c(1:31),lty=3)

#  fit best model for p=27 (AIC)

xvarm <- names(coef(regfit_bwd,id=27))[2:28]
housingbk.best <- z_plm[,c("SalePrice",xvarm)]
lmod <- lm(SalePrice~.,data=housingbk.best)
par(mfrow=c(2,2))
plot(lmod)
summary(lmod)
vif(lmod)

dim(housingbk.best)

#Splitting the dataset into train and test in the ratio 80:20
set.seed(1234)
smp_size <- floor(0.8 * nrow(housingbk.best))
train_ind=sample(seq_len(nrow(housingbk.best)), size = smp_size, rep=F)

train_data_bk <- housingbk.best[train_ind,]
test_data_bk <- housingbk.best[-train_ind,]

train_data_bk$SalePrice <- log(train_data_bk$SalePrice)

set.seed(1234)
lm_fit <- lm(SalePrice~.,data=train_data_bk)
summary(lm_fit)

lm_predict <- predict(lm_fit,test_data_bk)
msebk<-mean((test_data_bk$SalePrice-exp(lm_predict))^2)
sqrt(msebk)

#Best subset selection using Cross Validation
#Function to predict from reg subsets object
pred.sbs <- function(obj,new,id){
  form <- as.formula(obj$call[[2]])
  mat <- model.matrix(form,new)
  coefi <- coef(obj,id=id)
  xvars <- names(coefi)
  return(mat[,xvars]%*%coefi)
}

#Prep for cross validation(cv)
k <- 5  # set number of folds
p <- 31 # number of predictor variables

RNGkind(sample.kind = "Rounding")
set.seed(1234)

#Create an index with id 1-5 to assign observations to folds
folds <- sample(1:k, nrow(z_plm),replace=T)
folds

#Create dummy matrix to store CV error estimates
cv.err <- matrix(NA,k,p,dimnames=list(NULL,paste(1:p)))
cv.err

#Performing CV
for (j in 1:k){
  #Pick models with lowest RSS with 1-31 predictors fit without kth fold
  best.mods <- regsubsets(SalePrice~.,data=z_plm[folds!=j,], nvmax=31,method="exhaustive")
  #Estimate test error for all 31 models by predicting kth fold 
  for (i in 1:p){
    pred <- pred.sbs(best.mods,z_plm[folds==j,],id=i)
    cv.err[j,i] <- mean((z_plm$SalePrice[folds==j]-pred)^2)
  }
}
cv.err

mse.cv <- apply(cv.err,2,mean) # compute mean MSE for each number of predictors
mse.cv
min <- which.min(mse.cv)  # find minimum mean MSE
min
oneSE.cv <- apply(cv.err,2,sd) # compute standard error for each number of predictors
min1se <- mse.cv[min]+oneSE.cv[min]
min1se
mse.cv[min]

min.1se = 0
# find 1se number of predictors
for(i in 1:p){
  if(mse.cv[i]>min1se){
    min.1se <- i+1
  }
}
min
min.1se
mse.cv[min.1se]

# plot and put a red circle around lowest MSE, blue circle around 1se MSE
par(mfrow=c(1,1))
plot(1:31,mse.cv,type="b",xlab="no. of predictors)",ylab="est. test MSE",pch=19)
points(min,mse.cv[min],cex=2,col="red",lwd=2)
points(min.1se,mse.cv[min.1se],cex=2,col="blue",lwd=2)
abline(h=min1se,lty=2,col="blue") # plot 1se line

# Fit model for p=24

lmod.cv <- train(SalePrice~.,data=train_data_plm,method="lm", 
                 trControl=trainControl(method="repeatedcv",number=5,repeats=10))
summary(lmod.cv)
pred2 <- predict(lmod.cv,test_data_plm)
mse = mean((test_data_plm$SalePrice-exp(pred2))^2)
sqrt(mse)
lmod.cv$results[2]
mse.cv <- lmod.cv$results[2]^2

# Fitting regression tree model to train data
RNGkind(sample.kind = "Rounding")
set.seed(1234)
tree.mod <- rpart(SalePrice~.,method="anova", data=train_data_lm,
                  minsplit=2,maxsurrogate=0)

#Plotting the tree
plot(tree.mod, uniform=T,main="Regression Tree for SalePrice")
text(tree.mod)

# Detailed summary of splits
summary(tree.mod) 

# Test error for the regression tree model
y_hat1 <- predict(tree.mod, newdata = test_data_lm)
test.MSE1 <- mean((exp(y_hat1) - test_data_lm$SalePrice)^2)
sqrt(test.MSE1)

# Cross validation for trees
printcp(tree.mod) 
plotcp(tree.mod)

#Pruning the tree
pfit <- prune(tree.mod, cp=tree.mod$cptable[9])
plot(pfit, uniform=T, main="Pruned Regression Tree for Sale Price")
text(pfit)

# Test error of the pruned tree
y_hat2 <- predict(pfit, newdata = test_data_lm)
test.MSE2 <- mean((exp(y_hat2) - test_data_lm$SalePrice)^2)
sqrt(test.MSE2)

# Bagging method
RNGkind(sample.kind = "Rounding")
set.seed(1234)
bag.mod <- randomForest(SalePrice~.,data=train_data_lm,mtry=10,ntree=2000,importance=T)
bag.mod

plot(bag.mod)

# Importance of variables in bagging method
varImpPlot(bag.mod,type=1,pch=19)

# Test error of bagging model
y_hat3 <- predict(bag.mod, newdata = test_data_lm)
test.MSE3 <- mean((exp(y_hat3) - test_data_lm$SalePrice)^2)
sqrt(test.MSE3)

# Random forest model

control <- trainControl(method="cv", number=5, search="grid")
RNGkind(sample.kind = "Rounding")
set.seed(1234)
tunegrid <- expand.grid(mtry=c(1:31))
rf_gridsearch <- train(SalePrice~.,data=train_data_plm, method="rf", metric="RMSE", 
                       tuneGrid=tunegrid, trControl=control)
print(rf_gridsearch)

plot(rf_gridsearch)

# Test error for Random forest
y_hat4 <- predict(rf_gridsearch, newdata = test_data_plm)
test.MSE4 <- mean((exp(y_hat4) - test_data_lm$SalePrice)^2)
sqrt(test.MSE4)

# Gradient boosting
control <- trainControl(method="cv", number=5, search="grid")
RNGkind(sample.kind = "Rounding")
set.seed(1234)
tunegrid <- expand.grid(n.trees=c(100,500,1000,2000,5000,7500),
                        interaction.depth=c(1,3,5),
                        shrinkage=c(0.001,0.005,0.01),
                        n.minobsinnode=c(1,3,5))
gb_gridsearch <- train(SalePrice~.,data=train_data_plm, 
                       method="gbm", metric="RMSE", 
                       tuneGrid=tunegrid, trControl=control)
print(gb_gridsearch)

plot(gb_gridsearch)

#Fit using optimal parameters   
RNGkind(sample.kind = "Rounding")
set.seed(1234)
gb.mod <- gbm(SalePrice~.,data=train_data_plm, 
              distribution = "gaussian",n.trees = 5000,
              shrinkage = 0.005, interaction.depth = 3, 
              n.minobsinnode=1)

summary(gb.mod,cBars=20)

# Test error for Random forest
y_hat5 <- predict(gb.mod, newdata = test_data_plm, n.trees = 5000)
test.MSE5 <- mean((exp(y_hat5) - test_data_plm$SalePrice)^2)
sqrt(test.MSE5)

