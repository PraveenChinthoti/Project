
# Load the dataset
data <- read.csv(file.choose())

null_count <- sapply(data, function(x) sum(is.na(x)))
null_count

#Histogram and Density Plots
#Show the distribution of individual features.
hist(data$age, main="Age Distribution", xlab="Age")

#Compare distributions of numerical features across different groups, especially by DEATH_EVENT.
boxplot(data$ejection_fraction ~ data$DEATH_EVENT,
        main="Ejection Fraction by Death Event",
        xlab="Death Event",
        ylab="Ejection Fraction")

#Correlation Heatmap
#Show the correlations between numerical features, which can indicate multicollinearity.
#install.packages("ggcorrplot")
library(ggplot2)
library(ggcorrplot)
corr <- cor(data[, sapply(data, is.numeric)])
ggcorrplot(corr, method="circle", type="lower", lab=TRUE)

#Violin Plot
#Display distributions and density of features such as age or platelets by DEATH_EVENT.

ggplot(data,
       aes(x=factor(DEATH_EVENT),
           y=age,
           fill=factor(DEATH_EVENT))) +
  geom_violin()+
  labs(
    x="Death Event",
    y="Age")


library(caTools)
set.seed(123)  
split <- sample.split(data$DEATH_EVENT, SplitRatio = 0.7)
train <- subset(data, split == TRUE)
train
test <- subset(data, split == FALSE)
test


########################## applying Linear Regression ##########################
model <- lm(DEATH_EVENT ~ ., data = train)
model

predictions <- predict(model, test)
predictions

# Convert predictions to binary outcomes (threshold = 0.5)
binary_predictions <- ifelse(predictions > 0.5, 1, 0)
binary_predictions

accuracy <- mean(binary_predictions == test$DEATH_EVENT)
accuracy
print(paste("Accuracy:", round(accuracy * 100, 2), "%"))

library(ggplot2)

# Create a data frame for ggplot
plot_data <- data.frame(
  Actual = test$DEATH_EVENT,
  Predicted = predictions
)

# Plot actual vs. predicted values with ggplot2
ggplot(plot_data, aes(x = accuracy, y = predictions)) +
  geom_point(color = "blue", alpha = 0.6) +
  geom_jitter(width = 0.1, height = 0, color = "blue", alpha = 0.5) +
  geom_hline(yintercept = 0.5, color = "red", linetype = "dashed") +
  labs(title = "Actual vs Predicted DEATH_EVENT",
       x = "Actual DEATH_EVENT",
       y = "Predicted Probability") +
  theme_minimal()


#install.packages("pROC")
library(pROC)

# Compute ROC curve
roc_curve <- roc(response = test$DEATH_EVENT, predictor = probs, levels = c(0, 1), direction = "<")

# Suppress informational messages
#suppressMessages({
#  roc_curve <- roc(test$DEATH_EVENT, predictions)
#})

library(ggplot2)
ggroc(roc_curve, color = "blue", size = 1) +
  labs(title = "ROC Curve for Linear Regression Model",
       x = "False Positive Rate (1 - Specificity)",
       y = "True Positive Rate (Sensitivity)") +
  theme_minimal() +
  annotate("text", x = 0.75, y = 0.25,
           label = paste("AUC =", round(auc(roc_curve), 3)),
           color = "blue", size = 5)

auc_value <- auc(roc_curve)
print(paste("AUC:", round(auc_value, 3)))


################### DT ##############################################
library(caTools)
library(rpart)
library(rpart.plot)

dt_model <- rpart(DEATH_EVENT ~ ., data = train, method = "class")

rpart.plot(dt_model, type = 3, extra = 101, under = TRUE, fallen.leaves = TRUE)

probs <- predict(dt_model, test, type = "prob")[, 2]  # Get probabilities for DEATH_EVENT = 1

accuracy <- mean(binary_predictions == test$DEATH_EVENT)
print(paste("Accuracy:", round(accuracy * 100, 2), "%"))

# Compute the ROC curve
roc_curve <- roc(test$DEATH_EVENT, probs)
roc_curve <- roc(response = test$DEATH_EVENT, predictor = probs, levels = c(0, 1), direction = "<")


# Plot the ROC curve
plot(roc_curve, col = "blue", main = "ROC Curve for Decision Tree Model")
abline(a = 0, b = 1, lty = 2, col = "red")  # Add a diagonal line for random guessing

# Display AUC
auc_value <- auc(roc_curve)
print(paste("AUC:", round(auc_value, 3)))



######################## SVM ###############################################

library(e1071)
library(caTools)

# Support Vector Machine with Linear Kernel
svm_linear <- svm(DEATH_EVENT ~ ., data = train, kernel = "linear", type = "C-classification" , probability = TRUE)
summary(svm_linear)

predictions_linear <- predict(svm_linear, test)
predictions_linear

accuracy_linear <- mean(predictions_linear == test$DEATH_EVENT)
print(paste("Accuracy for SVM with Linear Kernel:", round(accuracy_linear * 100, 2), "%"))

svm_rbf <- svm(DEATH_EVENT ~ ., data = train, kernel = "radial", type = "C-classification" , probability = TRUE)
summary(svm_rbf)

predictions_rbf <- predict(svm_rbf, test)
predictions_rbf

accuracy_rbf <- mean(predictions_rbf == test$DEATH_EVENT)
print(paste("Accuracy for SVM with Radial Kernel:", round(accuracy_rbf * 100, 2), "%"))

library(e1071)


# Get predicted probabilities for the positive class (DEATH_EVENT = 1) from both models
probs_linear <- attr(predict(svm_linear, test, probability = TRUE), "probabilities")[,2]
probs_rbf <- attr(predict(svm_rbf, test, probability = TRUE), "probabilities")[,2]

# Compute ROC curve for both models
roc_curve_linear <- roc(test$DEATH_EVENT, probs_linear, levels = c(0, 1), direction = "<")
roc_curve_rbf <- roc(test$DEATH_EVENT, probs_rbf, levels = c(0, 1), direction = "<")

# Plot ROC curves
plot(roc_curve_linear, col = "blue", main = "ROC Curve for SVM Models", lwd = 2)
lines(roc_curve_rbf, col = "red", lwd = 2)
legend("bottomright", legend = c("Linear Kernel", "Radial Kernel"),
       col = c("blue", "red"), lwd = 2)

# Display AUC for both models
auc_linear <- auc(roc_curve_linear)
auc_rbf <- auc(roc_curve_rbf)
print(paste("AUC for SVM with Linear Kernel:", round(auc_linear, 3)))
print(paste("AUC for SVM with Radial Kernel:", round(auc_rbf, 3)))



##################### KNN ###########################################

library(class)
library(ggplot2)
library(caTools)


# Prepare training and testing data
train_x <- train[, -ncol(train)]  # Exclude target column
train_y <- train$DEATH_EVENT
test_x <- test[, -ncol(test)]
test_y <- test$DEATH_EVENT



# Apply KNN with k = 5
pred_knn_5 <- knn(train = train_x, test = test_x, cl = train_y, k = 5)
accuracy_knn_5 <- mean(pred_knn_5 == test_y)
print(paste("Accuracy for KNN with k = 5:", round(accuracy_knn_5 * 100, 2), "%"))

# Apply KNN with k = 10
pred_knn_10 <- knn(train = train_x, test = test_x, cl = train_y, k = 10)
accuracy_knn_10 <- mean(pred_knn_10 == test_y)
print(paste("Accuracy for KNN with k = 10:", round(accuracy_knn_10 * 100, 2), "%"))

# Convert predictions to numeric for ROC analysis
pred_knn_5_numeric <- as.numeric(pred_knn_5) - 1  # KNN outputs factors, convert to binary (0, 1)
pred_knn_10_numeric <- as.numeric(pred_knn_10) - 1

# Compute ROC curves
roc_curve_knn_5 <- roc(test_y, pred_knn_5_numeric, levels = c(0, 1), direction = "<")
roc_curve_knn_10 <- roc(test_y, pred_knn_10_numeric, levels = c(0, 1), direction = "<")

# Plot ROC curves for both KNN models
ggroc(list(`k = 5` = roc_curve_knn_5, `k = 10` = roc_curve_knn_10)) +
  labs(title = "ROC Curve for KNN Models", x = "False Positive Rate", y = "True Positive Rate") +
  theme_minimal() +
  annotate("text", x = 0.7, y = 0.2, label = paste("AUC k=5:", round(auc(roc_curve_knn_5), 3)), color = "blue") +
  annotate("text", x = 0.7, y = 0.1, label = paste("AUC k=10:", round(auc(roc_curve_knn_10), 3)), color = "red")

# Accuracy Plot using ggplot2
accuracy_data <- data.frame(
  Model = factor(c("KNN (k=5)", "KNN (k=10)")),
  Accuracy = c(accuracy_knn_5, accuracy_knn_10)
)

ggplot(accuracy_data, aes(x = Model, y = Accuracy, fill = Model)) +
  geom_bar(stat = "identity", color = "black") +
  labs(title = "Accuracy Comparison for KNN Models", y = "Accuracy") +
  theme_minimal()



##################### NB ##################
library(e1071)
library(ggplot2)
library(caTools)

nb_default <- naiveBayes(DEATH_EVENT ~ ., data = train)
pred_nb_default <- predict(nb_default, test, type = "class")
accuracy_nb_default <- mean(pred_nb_default == test$DEATH_EVENT)
print(paste("Accuracy for Naïve Bayes (default):", round(accuracy_nb_default * 100, 2), "%"))


probs_nb_default <- predict(nb_default, test, type = "raw")[,2]  # Probability for DEATH_EVENT = 1

roc_curve_nb_default <- roc(test$DEATH_EVENT, probs_nb_default, levels = c(0, 1), direction = "<")

# Apply Naïve Bayes with Laplace smoothing
nb_laplace <- naiveBayes(DEATH_EVENT ~ ., data = train, laplace = 1)
pred_nb_laplace <- predict(nb_laplace, test, type = "class")
accuracy_nb_laplace <- mean(pred_nb_laplace == test$DEATH_EVENT)
print(paste("Accuracy for Naïve Bayes (Laplace = 1):", round(accuracy_nb_laplace * 100, 2), "%"))


probs_nb_laplace <- predict(nb_laplace, test, type = "raw")[,2]


roc_curve_nb_laplace <- roc(test$DEATH_EVENT, probs_nb_laplace, levels = c(0, 1), direction = "<")

# Plot ROC curves for both Naïve Bayes models
ggroc(list(`Default` = roc_curve_nb_default, `Laplace = 1` = roc_curve_nb_laplace)) +
  labs(title = "ROC Curve for Naïve Bayes Models", x = "False Positive Rate", y = "True Positive Rate") +
  theme_minimal() +
  annotate("text", x = 0.7, y = 0.2, label = paste("AUC Default:", round(auc(roc_curve_nb_default), 3)), color = "blue") +
  annotate("text", x = 0.7, y = 0.1, label = paste("AUC Laplace:", round(auc(roc_curve_nb_laplace), 3)), color = "red")


# Accuracy Plot using ggplot2
accuracy_data <- data.frame(
  Model = factor(c("Naïve Bayes (Default)", "Naïve Bayes (Laplace = 1)")),
  Accuracy = c(accuracy_nb_default, accuracy_nb_laplace)
)

ggplot(accuracy_data, aes(x = Model, y = Accuracy, fill = Model)) +
  geom_bar(stat = "identity", color = "black") +
  labs(title = "Accuracy Comparison for Naïve Bayes Models", y = "Accuracy") +
  theme_minimal()
