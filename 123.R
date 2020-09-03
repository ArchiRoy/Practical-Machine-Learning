library(caret)
library(rpart)
library(rpart.plot)
library(RColorBrewer)
library(rattle)
library(randomForest)
library(ggplot2)
library(corrplot)
library(gbm)

##getting and exploring the data

train_in <- read.csv("https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv", header=TRUE)
valid_in <- read.csv("https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv", header=TRUE)

dim(valid_in)
dim(train_in)

##cleaning the input data

trainData<- train_in[, colSums(is.na(train_in)) == 0]
validData <- valid_in[, colSums(is.na(valid_in)) == 0]
dim(trainData)
dim(validData)
#We now remove the first seven variables as they have little impact on the outcome classe
trainData <- trainData[, -c(1:7)]
validData <- validData[, -c(1:7)]
dim(trainData)
dim(validData)

##preparing the data set for prediction by splitting the training data into 70% as train data and 30% as test data. This splitting will server also to compute the out-of-sample errors.

##The test data renamed: valid_in (validate data) will stay as is and will be used later to test the prodction algorithm on the 20 cases

set.seed(1234) 
inTrain <- createDataPartition(trainData$classe, p = 0.7, list = FALSE)
trainData <- trainData[inTrain, ]
testData <- trainData[-inTrain, ]
dim(trainData)
dim(testData)

##Cleaning even further by removing the variables that are near-zero-variance

NZV <- nearZeroVar(trainData)
trainData <- trainData[, -NZV]
testData  <- testData[, -NZV]
dim(trainData)
dim(testData)

##After this cleaning we are down now to 53 variables

cor_mat <- cor(trainData[, -53])
corrplot(cor_mat, order = "FPC", method = "color", type = "upper", 
         tl.cex = 0.8, tl.col = rgb(0, 0, 0))

#In the corrplot graph the correlated predictors (variables ) are those with a dark color intersection.

##We now obtain the names of the highly correlated attributes

highlyCorrelated = findCorrelation(cor_mat, cutoff=0.75)
names(trainData)[highlyCorrelated]

##Model Building

#1.Prediction with classification trees

set.seed(12345)
decisionTreeMod1 <- rpart(classe ~ ., data=trainData, method="class")
fancyRpartPlot(decisionTreeMod1)
#We then validate the model "decisionTreeModel" on the testData to find out how well it performs by looking at the accuracy variable.
predictTreeMod1 <- predict(decisionTreeMod1, testData, type = "class")
cmtree <- confusionMatrix(predictTreeMod1, testData$classe)
cmtree
#Plot the matrix results
plot(cmtree$table, col = cmtree$byClass, 
     main = paste("Decision Tree - Accuracy =", round(cmtree$overall['Accuracy'], 4)))
#We see that the accuracy rate of the model is low: 0.6967 and therefore the out-of-sample-error is about 0.3 which is considerable.

##2.Prediction with random forest

#We fisrt determine the model
controlRF <- trainControl(method="cv", number=3, verboseIter=FALSE)
modRF1 <- train(classe ~ ., data=trainData, method="rf", trControl=controlRF)
modRF1$finalModel
#We shall now validate the model on the test data to see how well it performs by looking at the Accuracy variable
predictRF1 <- predict(modRF1, newdata=testData)
cmrf <- confusionMatrix(predictRF1, testData$classe)
cmrf
##The accuracy rate using the random forest is very high: Accuracy : 1 and therefore the out-of-sample-error is equal to 0. But it might be due to overfitting.
##We shall now plot the model
plot(modRF1)
plot(cmrf$table, col = cmrf$byClass, main = paste("Random Forest Confusion Matrix: Accuracy =", round(cmrf$overall['Accuracy'], 4)))

##Prediction with generalized boosted prediction model
set.seed(12345)
controlGBM <- trainControl(method = "repeatedcv", number = 5, repeats = 1)
modGBM  <- train(classe ~ ., data=trainData, method = "gbm", trControl = controlGBM, verbose = FALSE)
modGBM$finalModel
## A gradient boosted model with multinomial loss function.
## 150 iterations were performed.
## There were 52 predictors of which 42 had non-zero influence.
print(modGBM)
#validating the GBM model
predictGBM <- predict(modGBM, newdata=testData)
cmGBM <- confusionMatrix(predictGBM, testData$classe)
cmGBM
#The accuracy rate using the random forest is very high: Accuracy : 0.9736 and therefore the out-of-sample-error is equal to 0.0264

##Applying the best model to the validation data
#Comparing all the methods we see that Random Forest is the best method. 
Results <- predict(modRF1, newdata=validData)
Results
