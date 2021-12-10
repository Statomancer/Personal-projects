
# These are the packages that we're using in the project
library(caret) 
library(readr)
library(gridExtra)
library(ltm)
library(tidyverse)


set.seed(1997)
y <- rbinom(1000, 1, 0.95) # the parameter p is here set to 0.95, i.e. this is for the 'imbalanced' data set
x1 = matrix(c(0), nrow = 1000)
x2 = matrix(c(0), nrow = 1000)
x3 = matrix(c(0), nrow = 1000)

for (j in 1:length(y)){
  if(y[j] == 1) {
    x1[j] = rnorm(1, 1, 1)
    x2[j] = rpois(1, 17)
    x3[j] = rexp(1, 5)
  }else{
    x1[j] = rnorm(1, 0, 1)
    x2[j] = rpois(1, 15)
    x3[j] = rexp(1, 8)
  }
}

biserial.cor(x1, y, level = 2) # Point Biseral correlation for categorical and continuous variable

cor.test(x2, y, method = "kendall") # Kendall's Tau for categorical and discrete variable

biserial.cor(x3, y, level = 2) # Point Biseral correlation for categorical and continuous variable

simdata = as.data.frame(matrix(c(y, x1, x2, x3), ncol = 4, nrow = length(y)))

simdata$V1 = as.factor(simdata$V1)

# Incoming densities
plot21 = simdata %>%
  rename(Values = V1) %>%
  ggplot() +
  geom_density(mapping = aes(x = V2, fill = Values), 
               size = 0.8,
               alpha = 0.7) +
  labs(title = "Distribution of X1",
       subtitle = "Densitities of X1 for p=0.95",
       caption = "Point-biseral correlation = 0.1928534",
       x = "X1",
       y = "Density") +
  theme(plot.title = element_text(face = "bold"), 
        plot.background = element_rect(fill = "gray 92"),
        legend.background = element_rect(fill = "gray 92"))

plot22 = simdata %>%
  rename(Values = V1) %>%
  ggplot() +
  geom_density(mapping = aes(x = V3, fill = Values), 
               size = 0.8,
               alpha = 0.7) +
  labs(title = "Distribution of X2",
       subtitle = "Densitities of X2 for p=0.95",
       caption = "Kendall's Tau = 0.1368459",
       x = "X2",
       y = "Density") +
  theme(plot.title = element_text(face = "bold"), 
        plot.background = element_rect(fill = "gray 92"),
        legend.background = element_rect(fill = "gray 92"))

plot23 = simdata %>%
  rename(Values = V1) %>%
  ggplot() +
  geom_density(mapping = aes(x = V4, fill = Values), 
               size = 0.8,
               alpha = 0.7) +
  labs(title = "Distribution of X3",
       subtitle = "Densitities of X3 for p=0.95",
       caption = "Point-biseral correlation = 0.1027436",
       x = "X3",
       y = "Density") +
  theme(plot.title = element_text(face = "bold"), 
        plot.background = element_rect(fill = "gray 92"),
        legend.background = element_rect(fill = "gray 92"))


grid.arrange(plot11, plot21, plot12, plot22, plot13, plot23) # Prints the plots in a 3x2 grid


R = 1000 # We set R = 1000 in order to obtain 1000 data sets 
acc = numeric(R) # and to account for sampling variability
spe = numeric(R) # We create empty vectors in order to store the
set.seed(1997)   # 1000 evaluation metrics
for(i in 1:R){
  y <- rbinom(1000, 1, 0.5)   # Simulate 1000 y's from binomial distribution 
  x1 = matrix(c(0), nrow = 1000) # Create empty vectors for 
  x2 = matrix(c(0), nrow = 1000) # the explanatory variables
  x3 = matrix(c(0), nrow = 1000)
  
  for (j in 1:length(y)){ # Simulate the explanatory variables 
    if(y[j] == 1) {       # depending on the value of the 
      x1[j] = rnorm(1, 1, 1) # response variable, y
      x2[j] = rpois(1, 17)
      x3[j] = rexp(1, 5)
    }else{
      x1[j] = rnorm(1, 0, 1)
      x2[j] = rpois(1, 15)
      x3[j] = rexp(1, 8)
    }
  }
  
  data = as.data.frame(cbind(y, x1, x2, x3)) # We create a matrix containing our explanatory 
                                             # variables and the response variable
  data$y = as.factor(data$y)
  
  training.samples = createDataPartition(data$y, p=0.7, list=FALSE)
  train.data = data[training.samples, ] # We use a 70-30 split in order to divide
  test.data = data[-training.samples, ] # the data into a training and test set


# Support vector machine using 5-fold CV and the radial kernel
  svm <- train(y~., 
               data = train.data,
               method = "svmRadial", 
               trControl = trainControl("cv", number = 5), 
               tuneLength = 10, 
               preProcess = c("center", "scale")
  )
  
  predicted = predict(svm, test.data)  # Use the model to predict the response variable in the test set
  acc[i] = mean(predicted == test.data$y) # Save the 1000 accuracies in the vector "acc".
  spe[i] = sensitivity(predicted, test.data$y, positive = "0") # We save the specificities in the 
}                                                          # "spe" vector using the sensitivity() function
                                                           # but changing the class which the function treats as being positive

mean.acc = mean(acc)  # Calculate the mean of the 1000 accuracies
mean.spe = mean(spe)  # Calculate the mean of the 1000 specificities

acc = as.data.frame(acc)
spe = as.data.frame(spe)

ci_acc = matrix(nrow = 2, ncol = 1) # Create empty vectors to save the upper and lower bound for the CI 
ci_spe = matrix(nrow = 2, ncol = 1) # - || -

ci_acc[1,1] = sort(acc[,1])[round(0.05/2 * length(acc[,1]))]  # Calculating the lower bound for the 95% CI
ci_acc[2,1] = sort(acc[,1])[round((1-0.05/2) * length(acc[,1]))] # - || - upper bound for the 95% CI

ci_spe[1,1] = sort(spe[,1])[round(0.05/2 * length(spe[,1]))] # Same as above but for the specificity
ci_spe[2,1] = sort(spe[,1])[round((1-0.05/2) * length(spe[,1]))]

Metric = numeric(R) # Creating empty vectors
mean = numeric(R)   
lower = numeric(R)
upper = numeric(R)

Acc = cbind(Metric, acc, mean, lower, upper) # Creating a matrix by combining the vectors

Acc = Acc %>%
  rename(Values = acc) # Renaming the variable acc to Values

for(i in 1:R){  # for-loop for input of entries into the matrix
  Acc$Metric[i] = "Accuracy"
  Acc$mean[i] = mean.acc
  Acc$lower[i] = ci_acc[1,1]
  Acc$upper[i] = ci_acc[2,1]
}


Spe = cbind(Metric, spe, mean, lower, upper) # Doing the same as above but for the specificity

Spe = Spe %>%
  rename(Values = spe)
  

for(i in 1:R){
  Spe$Metric[i] = "Specificity"
  Spe$mean[i] = mean.spe
  Spe$lower[i] = ci_spe[1,1]
  Spe$upper[i] = ci_spe[2,1]
}

joined = rbind(Acc, Spe) # joining the two matricies, in order for easier plotting


ci_plot = joined %>%  # Creating a plot containing all the values and confidence 
  ggplot(aes(group = Metric))+   # intervals for the evaluation metrics
  geom_errorbar(mapping = aes(y = Values, x = Metric, ymin = lower,
                              ymax = upper),
                alpha = 0.5,
                size = 0.5,
                width = 0.2)+
  geom_point(mapping = aes(y = Values, x = Metric, fill = Metric, colour = Metric),
             alpha = 0.5,
             size = 0.5)+
  geom_point(mapping = aes(y = mean, x = Metric),
             size = 3)+
  labs(title = "Simulation - Evalutation metrics",
       subtitle = "Accuracy and Specificity - Point represents the mean",
       y = "Metric values",
       caption = "P(Success) = 0.5")+
  theme(plot.title = element_text(face="bold", size = 16),
        plot.subtitle = element_text(size = 14),
        plot.background = element_rect(fill = 'gray 92'),
        legend.background = element_rect(fill = 'gray 92'),
        legend.text = element_text(size = 12),
        legend.title = element_text(face="bold", size = 14),
        axis.title = element_text(size = 14),
        axis.text.x = element_text(size = 12))+
  scale_y_continuous(n.breaks = 10)+
  facet_wrap(vars(Metric), labeller = "label_both", scales = "free")

ci_plot # the code above is used twice in order to get the code for when p = 0.5 and when p = 0.95

grid.arrange(ci_plot, ci_plot1) # We 



##### Simulation
library(ltm)

set.seed(1997)
y <- rbinom(1000, 1, 0.5) # the parameter p is here set to 0.5, i.e. this is for the 'balanced' data set
x1 = matrix(c(0), nrow = 1000)
x2 = matrix(c(0), nrow = 1000)
x3 = matrix(c(0), nrow = 1000)

for (j in 1:length(y)){
  if(y[j] == 1) {
    x1[j] = rnorm(1, 1, 1)
    x2[j] = rpois(1, 17)
    x3[j] = rexp(1, 5)
  }else{
    x1[j] = rnorm(1, 1, 1)
    x2[j] = rpois(1, 15)
    x3[j] = rexp(1, 8)
  }
}


biserial.cor(x1, y, level = 2) # Point Biseral correlation for categorical and continuous variable

cor.test(x2, y, method = "kendall") # Kendall's Tau for categorical and discrete variable

biserial.cor(x3, y, level = 2) # Point Biseral correlation for categorical and continuous variable

simdata = as.data.frame(matrix(c(y, x1, x2, x3), ncol = 4, nrow = length(y)))

simdata$V1 = as.factor(simdata$V1)

# Incoming densities
plot11 = simdata %>%
  rename(Values = V1) %>%
  ggplot() +
  geom_density(mapping = aes(x = V2, fill = Values), 
               size = 0.8,
               alpha = 0.7) +
  labs(title = "Distribution of X1",
       subtitle = "Densitities of X1 for p=0.5",
       caption = "Point-biseral correlation = 0.4403131",
       x = "X1",
       y = "Density") +
  theme(plot.title = element_text(face = "bold"), 
        plot.background = element_rect(fill = "gray 92"),
        legend.background = element_rect(fill = "gray 92"))

plot12 = simdata %>%
  rename(Values = V1) %>%
  ggplot() +
  geom_density(mapping = aes(x = V3, fill = Values), 
               size = 0.8,
               alpha = 0.7) +
  labs(title = "Distribution of X2",
       subtitle = "Densitities of X2 for p=0.5",
       caption = "Kendall's Tau = 0.20724",
       x = "X2",
       y = "Density") +
  theme(plot.title = element_text(face = "bold"), 
        plot.background = element_rect(fill = "gray 92"),
        legend.background = element_rect(fill = "gray 92"))

plot13 = simdata %>%
  rename(Values = V1) %>%
  ggplot() +
  geom_density(mapping = aes(x = V4, fill = Values), 
               size = 0.8,
               alpha = 0.7) +
  labs(title = "Distribution of X3",
       subtitle = "Densitities of X3 for p=0.5",
       caption = "Point-biseral correlation = 0.2186644",
       x = "X3",
       y = "Density") +
  theme(plot.title = element_text(face = "bold"), 
        plot.background = element_rect(fill = "gray 92"),
        legend.background = element_rect(fill = "gray 92"))






hitsong <- read.csv("~/Desktop/hitsong.csv", stringsAsFactors = TRUE) 
hitsong <- subset(hitsong, select = -c(track_title, artist_name,  # We omit the variables that we won't
                                      track_id, key, mode,        # use in the analysis
                                      duration_ms, time_signature))


hitsong$On_chart <- as.factor(hitsong$On_chart) # Make sure that R interprets the response variable as 
                                                # binary 

##### Exploratory data analysis

sum(is.na(hitsong)) # No missing values

hitsong %>%
  summarise(Mean_on_chart = mean(On_chart)) # Half of the observations are hit songs!

int_vec = matrix(c(0), nrow = 2, ncol = 10)
for (i in 1:ncol(int_vec)) {
  int_vec[1,i] = max(hitsong[,i])
  int_vec[2,i] = min(hitsong[,i])
}
int_vec # Finds the interval of the variables (max. and min values.)


hitsong %>%
  rename(Values = On_chart) %>%   # Boxplots
  ggplot() +
  geom_boxplot(mapping = aes(x = instrumentalness, fill = Values), 
               size = 0.8,
               alpha = 0.7)

y = hitsong$On_chart
x1 = hitsong$energy
x2 = hitsong$acousticness
x3 = hitsong$danceability
x4 = hitsong$instrumentalness   # Recoding for correlation coefficients
x5 = hitsong$liveness
x6 = hitsong$loudness
x7 = hitsong$speechiness
x8 = hitsong$valence
x9 = hitsong$tempo

biserial.cor(x1, y, level = 2)
biserial.cor(x2, y, level = 2)
biserial.cor(x3, y, level = 2)
biserial.cor(x4, y, level = 2)
biserial.cor(x5, y, level = 2)  # Point biserial correlation coefficients
biserial.cor(x6, y, level = 2)
biserial.cor(x7, y, level = 2)
biserial.cor(x8, y, level = 2)
biserial.cor(x9, y, level = 2)

# Incoming boxplots

plotX1 = hitsong %>%
  rename(Values = On_chart) %>%
  ggplot() +
  geom_boxplot(mapping = aes(x = energy, fill = Values), 
               size = 0.8,
               alpha = 0.7) +
  coord_flip() +
  theme(axis.title.x=element_blank(),
        axis.text.x=element_blank(),
        axis.ticks.x=element_blank())

plotX2 = hitsong %>%
  rename(Values = On_chart) %>%
  ggplot() +
  geom_boxplot(mapping = aes(x = acousticness, fill = Values), 
               size = 0.8,
               alpha = 0.7) +
  coord_flip() +
  theme(axis.title.x=element_blank(),
        axis.text.x=element_blank(),
        axis.ticks.x=element_blank())

plotX3 = hitsong %>%
  rename(Values = On_chart) %>%
  ggplot() +
  geom_boxplot(mapping = aes(x = danceability, fill = Values), 
               size = 0.8,
               alpha = 0.7) +
  coord_flip() +
  theme(axis.title.x=element_blank(),
        axis.text.x=element_blank(),
        axis.ticks.x=element_blank())

plotX4 = hitsong %>%
  rename(Values = On_chart) %>%
  ggplot() +
  geom_boxplot(mapping = aes(x = instrumentalness, fill = Values), 
               size = 0.8,
               alpha = 0.7) +
  coord_flip() +
  theme(axis.title.x=element_blank(),
        axis.text.x=element_blank(),
        axis.ticks.x=element_blank())

plotX5 = hitsong %>%
  rename(Values = On_chart) %>%
  ggplot() +
  geom_boxplot(mapping = aes(x = liveness, fill = Values), 
               size = 0.8,
               alpha = 0.7) +
  coord_flip() +
  theme(axis.title.x=element_blank(),
        axis.text.x=element_blank(),
        axis.ticks.x=element_blank())

plotX6 = hitsong %>%
  rename(Values = On_chart) %>%
  ggplot() +
  geom_boxplot(mapping = aes(x = loudness, fill = Values), 
               size = 0.8,
               alpha = 0.7) +
  coord_flip() +
  theme(axis.title.x=element_blank(),
        axis.text.x=element_blank(),
        axis.ticks.x=element_blank())

plotX7 = hitsong %>%
  rename(Values = On_chart) %>%
  ggplot() +
  geom_boxplot(mapping = aes(x = speechiness, fill = Values), 
               size = 0.8,
               alpha = 0.7) +
  coord_flip() +
  theme(axis.title.x=element_blank(),
        axis.text.x=element_blank(),
        axis.ticks.x=element_blank())

plotX8 = hitsong %>%
  rename(Values = On_chart) %>%
  ggplot() +
  geom_boxplot(mapping = aes(x = valence, fill = Values), 
               size = 0.8,
               alpha = 0.7) +
  coord_flip() +
  theme(axis.title.x=element_blank(),
        axis.text.x=element_blank(),
        axis.ticks.x=element_blank())

plotX9 = hitsong %>%
  rename(Values = On_chart) %>%
  ggplot() +
  geom_boxplot(mapping = aes(x = tempo, fill = Values), 
               size = 0.8,
               alpha = 0.7) +
  coord_flip() +
  theme(axis.title.x=element_blank(),
        axis.text.x=element_blank(),
        axis.ticks.x=element_blank())

grid.arrange(plotX1, plotX2, plotX3,
             plotX4, plotX5, plotX6, # Prints the plots in a 3x3 grid
             plotX7, plotX8, plotX9)



### Data preparation - dividing the data into a training set and test set with a 70-30 split

set.seed(1997)
training.samples = createDataPartition(hitsong$On_chart, p=0.7, list=FALSE)
train.data = hitsong[training.samples, ]
test.data = hitsong[-training.samples, ]



svm <- train(On_chart~.,  # Support vector machine using 5-fold CV and the radial kernel
             data = train.data,
             method = "svmRadial", 
             trControl = trainControl("cv", number = 5), 
             tuneLength = 10, 
             preProcess = c("center", "scale")
)

predicted = predict(svm, test.data)  # predicting the response variable using the model
mean(predicted == test.data$On_chart) # calculating the accuracy
conf_matrix = confusionMatrix(predicted, test.data$On_chart, positive = "1", mode = "everything")
conf_matrix  # Creating a confusion matrix with positive = 1




logit <- glm(On_chart~., data = train.data, family = "binomial")  # Use the glm()-function 

predicted_values = as.factor(ifelse(predict(logit, test.data, type="response")>0.5,1,0))
conf_matrix = confusionMatrix(predicted_values, as.factor(test.data$On_chart), positive = "1", mode = "everything")
conf_matrix    # confusion matrix for the predictions with positive = 1
summary(logit)



rf <- train(On_chart~.,    # Random forest with ntree = 5 and 5-fold CV for mtry
            data = train.data,
            method = "rf",
            ntree = 500,
            trControl = trainControl("cv", number = 5), 
            tuneLength = 10, 
            preProcess = c("center", "scale")
)

predicted = predict(rf, test.data)  # Using the model to predict the response variable
mean(predicted == test.data$On_chart) # Accuracy of the model
conf_matrix = confusionMatrix(predicted, test.data$On_chart, positive = "1", mode = "everything")
conf_matrix # Confusion matrix for the predictions with positive = 1


