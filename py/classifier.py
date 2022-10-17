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