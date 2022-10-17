# run from .../mc-take_home_task

import pandas as pd
import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler
    
class RansomwareClassifier:
    def __init__(self):
        pass

    def predict(self,X):
        return(self.predict(X))
    
    def score(self,X,y):
        return(self.model.score(X,y))
    
class LogisticRegressionRansomwareClassifier(RansomwareClassifier):
    pass

    def predict(self, X):
        # standardise the features
        scaled_X = pd.DataFrame(self.scaler.transform(X), columns=X.columns)
        
        return(self.model.predict(scaled_X))

    def train(self,X,y,penalty="none"):
        # standardise the features
        scaler = StandardScaler()
        scaler.fit(X)
        scaled_X = pd.DataFrame(scaler.transform(X), columns=X.columns)

        lr = LogisticRegression(max_iter=1000, penalty=penalty)
        
        self.scaler = scaler
        self.model = lr.fit(scaled_X,y)
        
# proportion to use for training models
prop_test = 0.25

# load the data
data = pd.read_csv("data/BitcoinHeistData.csv")

# add binary class
data["ransomware"] = (data["label"] != "white").astype(int)

# feature names
name_features = ["length", "weight", "count", "looped", "neighbors", "income"]

# outcome name
name_outcome = "ransomware"

# split data into training and test datasets
X_train, X_test, y_train, y_test = train_test_split(
    data[name_features].copy(), 
    data[name_outcome].copy(), 
    test_size=prop_test, 
    random_state=56465
)

# initialise the LogisticRegressionRansomwareClassifier
classifier = LogisticRegressionRansomwareClassifier()

# train the classifer
classifier.train(X_train, y_train)

# data properties
print("number of ransomware address", sum(data["ransomware"]))
print("data.shape:", data.shape)
print("number of missing values:", sum(data.isna().sum()))

# training data properties
print("number of observations:", X_train.shape[0])
print("number of ransomware addresses", sum(y_train))

# test data properties
print("number of observations:", X_test.shape[0])
print("number of ransomware addresses", sum(y_test))

# evaluate performance
predictions = classifier.predict(X_test)
print("number predicted ransomdeware addresses:", sum(predictions))
print("true value of class where address label predicted to by ransomware")
print(y_test.loc[y_test.index[np.where(predictions)]])

# model coefficients
print(
    pd.DataFrame(
        {
            "feature_name" : list(classifier.model.feature_names_in_) + ["intercept"],
             "coef" : list(classifier.model.coef_.squeeze()) + list(classifier.model.intercept_)
        }
    )
)

# confusion matrix
print(confusion_matrix(y_test, classifier.predict(X_test)).ravel())