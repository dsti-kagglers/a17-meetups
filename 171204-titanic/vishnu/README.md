# Kaggle-Titanic
Titanic Survival analysis in R
#--setwd("C:/Users/Welcome/Desktop/CHAITU/kaggle/Titanic")

Train_data<-read.csv(file='train.csv',header=TRUE,stringsAsFactors = FALSE)
head(Train_data)

Test_data<-read.csv('test.csv',stringsAsFactors = FALSE,header=TRUE)

head(Test_data)


Train_data$IsTrainSet<-TRUE
Test_data$IsTrainSet<-FALSE

names(Train_data)
names(Test_data)

Test_data$Survived<-NA


Titanic<-rbind(Train_data,Test_data)

head(Titanic)

table(Titanic$IsTrainSet)


table(Titanic$Embarked)

Titanic[Titanic$Embarked=='','Embarked']<-'S'



table(Titanic$Embarked)

#Fill Missing Values of Fare
age_median<-median(Titanic$Age,na.rm =TRUE)
age_median

Titanic$Age[is.na(Titanic$Age)]<-age_median


Titanic$Age[is.na(Titanic$Age)]

#-----------------------------
#-------------------------------Fill missing values of Fare------------
table(is.na(Titanic$Fare))
Fare_median=median(Titanic$Fare,na.rm=TRUE)
Titanic[is.na(Titanic$Fare),'Fare']<-Fare_median
 # Categorical Casting of pclass,sex,embarked

Titanic$Pclass<-as.factor(Titanic$Pclass)
Titanic$Sex<-as.factor(Titanic$Sex)
Titanic$Embarked<-as.factor(Titanic$Embarked)

#seperate the Train and Test Data Set

Train_data<-Titanic[Titanic$IsTrainSet==TRUE,]
Test_data<-Titanic[Titanic$IsTrainSet==FALSE,]

str(Titanic)
head(Titanic[Titanic$IsTrainSet==TRUE,])

str(Train_data)
Train_data$Survived<-as.factor(Train_data$Survived)

#install Random forest pacake and call it
install.packages("randomForest")
library(randomForest)
equation<-'Survived ~Pclass+Age+Sex+SibSp+Parch+Embarked+Fare'
sur_eqa<-formula(equation)

Titanic_sur<-randomForest(formula=sur_eqa,data=Train_data,ntree=500,mtry=3,nodesize=0.01*nrow(Test_data))

Survived<-predict(Titanic_sur,newdata = Test_data)
Survived
result_set<-data.frame(PassengerId=Test_data$PassengerId,Survived=Survived)


write.csv(result_set,file="my_solution4.csv",row.names=FALSE)
