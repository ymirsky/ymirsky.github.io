import pandas as pd
import numpy as np


from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score


class LoanPredictor:
    def __init__(self):
        print("Setting up Victim Model")
        datapath = "loan_dataset.csv"
        D = pd.read_csv(datapath)
        X = D[D.columns[1:-1]].to_numpy()
        Y = D[D.columns[-1]].to_numpy()
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.20, random_state=42)

        M = RandomForestClassifier(n_estimators=10)
        self.M = M.fit(X_train,Y_train)
        self.X_test = X_test
        self.Y_test = Y_test

        Y_pred = M.predict(X_test)
        self.ACC = np.round(accuracy_score(Y_test, Y_pred),3)
        self.AUC = np.round(roc_auc_score(Y_test, Y_pred),3)

    def predict(self, x):
        return self.M.predict(x)

    #checks a model's performance against M
    # model must have a predict() method that accepts bx3 arrays (b:batch)
    def check_model_performance(self, M_t):
        Y_pred = M_t.predict(self.X_test)
        ACC = np.round(accuracy_score(self.Y_test, Y_pred), 3)
        AUC = np.round(roc_auc_score(self.Y_test, Y_pred), 3)
        print("Loan Predictor, Accuracy:", self.ACC, "AUC:", self.AUC)
        print("Stolen Model, Accuracy:", ACC, "AUC:", AUC)