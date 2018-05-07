import pandas as pd
import logging

from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

def myfunc():
    print("Hello module!")


def loadingData(dataPath='../../data/', logLevel="DEBUG"):

    # this block is logger initialisation
    #TODO: put it in a function at next use
    logger = logging.getLogger()
    logging.basicConfig(format='%(asctime)s :: %(levelname)s :: %(message)s')
    stream_handler = logging.StreamHandler()
    logger.setLevel(logLevel)
    logger.addHandler(stream_handler)

    logger.info('Loading and preparing Titanic data')
    logger.debug('pas bon')

    train_df = pd.read_csv(dataPath+"train.csv")
    print(train_df.head())

    train_df['AgeMed'] = train_df['Age'].fillna((train_df['Age'].median()))
    train_df['isAgeKnown'] = train_df['Age'].where(train_df["Age"].isnull(), 1).fillna(0).astype(int)


    train_df = pd.concat([train_df, pd.get_dummies(train_df["Sex"])], axis=1)
    train_df = pd.concat([train_df, pd.get_dummies(train_df["Embarked"].fillna("embarkedNotKnown"))], axis=1)
    train_df.rename(columns={'C': 'Cherbourg','Q': 'Queenstown', 'S': 'Southampton'}, inplace=True)
    train_df["FamilySize"]= train_df["SibSp"]+train_df["Parch"]
    def extract_Title(row):
        return row["Name"].split(', ' )[1].split(".")[0]
    train_df["Title"] = train_df.apply(extract_Title,axis=1)
    train_df = pd.concat([train_df, pd.get_dummies(train_df["Title"])], axis=1)

    columns = ["Fare", "Pclass","SibSp","Parch",
               "female","male",
               "AgeMed","isAgeKnown",
               "Cherbourg","Queenstown","Southampton",
               "FamilySize",
               "Master", "Miss", "Mr", "Mrs", "Ms", "Rev",
               ]

    labels = train_df["Survived"].values
    features = train_df[list(columns)].values

    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.3, random_state=0)
    tuned_parameters = [{'n_estimators': [10, 100, 1000],
                         'min_samples_split': [2,3]}]


    clf = GridSearchCV(RandomForestClassifier(), tuned_parameters, cv=10,
                       scoring='accuracy', n_jobs=-1)
    clf.fit(X_train, y_train)


    # In[61]:

    y_true, y_pred = y_test, clf.predict(X_test)
    print(classification_report(y_true, y_pred))
    print(accuracy_score(y_true, y_pred))

    return 1


if __name__ == '__main__':
    loadingData()
