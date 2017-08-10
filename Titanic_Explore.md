# Exploratory Data Analysis Of The Titanic Dataset
Kantesh Biswas  
August 4, 2017  



## Introduction
This markdown document is about exploratory data analysis on the Titanic dataset.
The first thing that i am going to do is some feature engineering.I am also going to make some really cool visualizations and at the end i am going to use **Random Forest** to make predictions.

The following section provides a brief description about all the columns in our train and test datasets.

Variable Name | Description
--------------|-------------
Survived      | Survived (1) or died (0)
Pclass        | Passenger's class
Name          | Passenger's name
Sex           | Passenger's sex
Age           | Passenger's age
SibSp         | Number of siblings/spouses aboard
Parch         | Number of parents/children aboard
Ticket        | Ticket number
Fare          | Fare
Cabin         | Cabin
Embarked      | Port of embarkation

## First thing first let's load all the libraries that we will be using.


```r
library(ggplot2)
library(dplyr)
library(mice)
library(randomForest)
```

Now that our packages are loaded let's load the datasets.


```r
train<-read.csv("train.csv",header = T,stringsAsFactors = F)
test<-read.csv("test.csv",header = T,stringsAsFactors = F)
combined<-bind_rows(train,test)
```
Here we have combined our train and test dataset because before starting model building we will do some feature engineering which will definately increase the accuracy of our model.
Since the datasets are loaded let's have a look at the structures of our datasets.


```r
str(train)
```

```
## 'data.frame':	891 obs. of  12 variables:
##  $ PassengerId: int  1 2 3 4 5 6 7 8 9 10 ...
##  $ Survived   : int  0 1 1 1 0 0 0 0 1 1 ...
##  $ Pclass     : int  3 1 3 1 3 3 1 3 3 2 ...
##  $ Name       : chr  "Braund, Mr. Owen Harris" "Cumings, Mrs. John Bradley (Florence Briggs Thayer)" "Heikkinen, Miss. Laina" "Futrelle, Mrs. Jacques Heath (Lily May Peel)" ...
##  $ Sex        : chr  "male" "female" "female" "female" ...
##  $ Age        : num  22 38 26 35 35 NA 54 2 27 14 ...
##  $ SibSp      : int  1 1 0 1 0 0 0 3 0 1 ...
##  $ Parch      : int  0 0 0 0 0 0 0 1 2 0 ...
##  $ Ticket     : chr  "A/5 21171" "PC 17599" "STON/O2. 3101282" "113803" ...
##  $ Fare       : num  7.25 71.28 7.92 53.1 8.05 ...
##  $ Cabin      : chr  "" "C85" "" "C123" ...
##  $ Embarked   : chr  "S" "C" "S" "S" ...
```

```r
str(test)
```

```
## 'data.frame':	418 obs. of  11 variables:
##  $ PassengerId: int  892 893 894 895 896 897 898 899 900 901 ...
##  $ Pclass     : int  3 3 2 3 3 3 3 2 3 3 ...
##  $ Name       : chr  "Kelly, Mr. James" "Wilkes, Mrs. James (Ellen Needs)" "Myles, Mr. Thomas Francis" "Wirz, Mr. Albert" ...
##  $ Sex        : chr  "male" "female" "male" "male" ...
##  $ Age        : num  34.5 47 62 27 22 14 30 26 18 21 ...
##  $ SibSp      : int  0 1 0 0 1 0 0 1 0 2 ...
##  $ Parch      : int  0 0 0 0 1 0 0 1 0 0 ...
##  $ Ticket     : chr  "330911" "363272" "240276" "315154" ...
##  $ Fare       : num  7.83 7 9.69 8.66 12.29 ...
##  $ Cabin      : chr  "" "" "" "" ...
##  $ Embarked   : chr  "Q" "S" "Q" "S" ...
```

```r
str(combined)
```

```
## 'data.frame':	1309 obs. of  12 variables:
##  $ PassengerId: int  1 2 3 4 5 6 7 8 9 10 ...
##  $ Survived   : int  0 1 1 1 0 0 0 0 1 1 ...
##  $ Pclass     : int  3 1 3 1 3 3 1 3 3 2 ...
##  $ Name       : chr  "Braund, Mr. Owen Harris" "Cumings, Mrs. John Bradley (Florence Briggs Thayer)" "Heikkinen, Miss. Laina" "Futrelle, Mrs. Jacques Heath (Lily May Peel)" ...
##  $ Sex        : chr  "male" "female" "female" "female" ...
##  $ Age        : num  22 38 26 35 35 NA 54 2 27 14 ...
##  $ SibSp      : int  1 1 0 1 0 0 0 3 0 1 ...
##  $ Parch      : int  0 0 0 0 0 0 0 1 2 0 ...
##  $ Ticket     : chr  "A/5 21171" "PC 17599" "STON/O2. 3101282" "113803" ...
##  $ Fare       : num  7.25 71.28 7.92 53.1 8.05 ...
##  $ Cabin      : chr  "" "C85" "" "C123" ...
##  $ Embarked   : chr  "S" "C" "S" "S" ...
```
Notice that in our test dataset there is no survived column because that's how kaggle works we need to predict the survival rate on the test dataset.
Let's check how many people have survived in our training dataset.

```r
table(train$Survived)
```

```
## 
##   0   1 
## 549 342
```
342 survived out of 549.Well we can also find the percentage of people that survived by using the following command.

```r
prop.table(table(train$Survived))
```

```
## 
##         0         1 
## 0.6161616 0.3838384
```
We can see that 38% of people has survived the disaster.That means most of the people abroad were perished.
Now let's make our first prediction,since most people died in our training set,perhaps we can assume that everyone in the test set also died.
So let's add our first prediction to the test set.

```r
test$Survived<-rep(0,418)
```
We need to submit a csv file with PassengerId and survived column to **Kaggle**.
So let's prepare our submission file.

```r
submit<-data.frame(PassengerId=test$PassengerId,Survived=test$Survived)
write.csv(submit,file = "Prediction_titanic.csv",row.names = FALSE)
```
Here we have excluded the row numbers oterwise kaggle will reject our submission.
As we know that at the time of the disaster women and children were saved first.Let's try to find that out.

```r
table(train$Sex)
```

```
## 
## female   male 
##    314    577
```
We can see that the majority of the passengers were male.Now let's find out the proportion of male and female passengers that survived the disaster.

```r
prop.table(table(train$Sex,train$Survived))
```

```
##         
##                   0          1
##   female 0.09090909 0.26150393
##   male   0.52525253 0.12233446
```
By default the proportion table command takes each entry in the table and divides by the total number of passengers.But we want to see the proportion of each sex that survived, as separate groups.We will pass one more argument **1** to our function which stands for the rows.**2** stands for columns.

```r
prop.table(table(train$Sex,train$Survived),1)
```

```
##         
##                  0         1
##   female 0.2579618 0.7420382
##   male   0.8110919 0.1889081
```
Well we can see that a very low percentage of male survived whereas majority of females abroad were survived.
Let's update our old prediction and mark all the females passenger in our test dataset as survived.

```r
test$Survived<-0
test$Survived[test$Sex=='female']<-1
```
Ok now we need to update our submission file also.

```r
submit<-data.frame(PassengerId=test$PassengerId,Survived=test$Survived)
write.csv(submit,file = "Prediction_titanic.csv",row.names = FALSE)
```
Now let's look into the age variable.

```r
summary(train$Age)
```

```
##    Min. 1st Qu.  Median    Mean 3rd Qu.    Max.    NA's 
##    0.42   20.12   28.00   29.70   38.00   80.00     177
```
There are 177 NA values.We will impute those NA values with the mean age of the rest of the passengers.
Let's create a new variable to find out if the passenger was below 18 or not.

```r
train$Child<-0
train$Child[train$Age<18]<-1
```
Let's find out the number of **Adults** and **Childrens** in our dataset by gender.
Aggregate function is very usefull when we want to apply a specific command for groups.

```r
aggregate(Survived~Child+Sex,data = train,FUN = length)
```

```
##   Child    Sex Survived
## 1     0 female      259
## 2     1 female       55
## 3     0   male      519
## 4     1   male       58
```
There are 259 female and 519 male adults in our dataset.Similarly 55 female and 58 male children are there.
Now we will find out how many male and female children survived the disaster.

```r
aggregate(Survived~Child+Sex,data = train,FUN = sum)
```

```
##   Child    Sex Survived
## 1     0 female      195
## 2     1 female       38
## 3     0   male       86
## 4     1   male       23
```
Here we can see **195** female **Adults** and **38** female children survived whereas **86** male **Adults** and **23** male children survived the disaster.










