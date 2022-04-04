import pandas as pd
import numpy as np


from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.neighbors import NearestNeighbors


class LoanPredictor:
    def __init__(self):
        print("Setting up Victim Model")
        datapath = "loan_dataset.csv"
        D = pd.read_csv(datapath)
        X = D[D.columns[1:-1]]
        Y = D[D.columns[-1]]
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.20, random_state=42)
        self.X_train = X_train.to_numpy()
        self.Y_train = Y_train.to_numpy()
        self.X_test = X_test.to_numpy()
        self.Y_test = Y_test.to_numpy()

        self.__train_model()

        Y_pred = self.M.predict(self.X_test)
        self.ACC = np.round(accuracy_score(self.Y_test, self.Y_pred),3)
        self.AUC = np.round(roc_auc_score(self.Y_test, self.Y_pred),3)

        # Setup poison detector
        self.norm = np.array([70000, 64, 14000])
        self.A = NearestNeighbors(n_neighbors=2).fit(self.X_train / self.norm)

    def predict(self, x):
        return self.M.predict(x)

    # X must be in format of bxn where b is the batch size
    def __check_for_poison(self, X, verbose=True):
        distances, samples = self.A.kneighbors(X/self.norm)
        threshold = 0.8
        if (np.mean(distances,axis=0) > threshold).any():
            if verbose:
                print("Poison Detected! Samples rejected")
            return None
        else:
            if verbose:
                print("Data is Safe. Added to training set")
            return X

    def __train_model(self):
        M = RandomForestClassifier(n_estimators=10)
        self.M = M.fit(self.X_train, self.Y_train)

    #checks a model's performance against M
    # model must have a predict() method that accepts bx3 arrays (b:batch)
    def check_model_performance(self, M_t):
        Y_pred = M_t.predict(self.X_test)
        ACC = np.round(accuracy_score(self.Y_test, Y_pred), 3)
        AUC = np.round(roc_auc_score(self.Y_test, Y_pred), 3)
        print("Loan Predictor, Accuracy:", self.ACC, "AUC:", self.AUC)
        print("Stolen Model, Accuracy:", ACC, "AUC:", AUC)

    #updates internal model with given data (checks for poison first)
    #data must be in bxn format where b is batch size, and in np.array format
    def update_model(self, X_train_extra, Y_train_extra, verbose=True):
        X_train_extra = self.__check_for_poison(X_train_extra, verbose)
        if X_train_extra is None:
            return
        self.X_train = np.vstack((self.X_train, X_train_extra))
        self.Y_train = np.concatenate((self.Y_train, Y_train_extra))
        self.__train_model()
        if verbose:
            print("Model updated")


