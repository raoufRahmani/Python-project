from dataclasses import dataclass 
import numpy as np 
import pandas as pd
from sklearn import datasets

# Partie 1


import numpy as np
import pandas as pd

class Dataset:
    def __init__(self, X: np.ndarray, Y: np.ndarray, features_names: list):
        self.X = X
        self.Y = Y
        self.features_names = features_names
        self.n = X.shape[0]
        self.p = X.shape[1]

    def add_intercept(self):
        self.X = np.column_stack([np.ones((self.n, 1)), self.X])
        self.features_names.insert(0, "constante")

    def to_dataframe(self):
        df = pd.DataFrame(self.X, columns=self.features_names)
        df["variable_expliquée"] = self.Y
        return df



# Partrie 2
@dataclass
class regression_lineaire:
    coefficients: np.ndarray
    features_names: list


def regression(X, Y, features_names):
    Xt_X = np.dot(X.T, X)
    Xt_Y = np.dot(X.T, Y)
    Beta = np.dot(np.linalg.inv(Xt_X), Xt_Y)
    intercept = Beta[0]
    betas = Beta[1:]
    return regression_lineaire(coefficients=Beta, features_names=features_names), intercept, betas


# 3eme partie : résultats
class results:
    def __init__(self, X, Y, df, Beta):
        self.X = X
        self.Y = Y
        self.df = df
        self.Beta = Beta
        self.valeurs_prédites = None
        self.erreurs = None
        self.R2 = None
        self.RMSE = None

    def predict(self):
        self.valeurs_prédites = np.dot(self.X, self.Beta)
        self.erreurs = self.Y - self.valeurs_prédites

    def metriques(self):
        ybar = np.mean(self.Y)
        SCT = np.sum((self.Y - ybar)**2) / len(self.Y)
        SCR = np.sum(self.erreurs**2) / len(self.Y)
        self.R2 = (SCT - SCR) / SCT
        self.RMSE = np.sqrt(SCR)

    def extend_df(self):
        self.df["valeurs_prédites"] = self.valeurs_prédites
        self.df["erreurs"] = self.erreurs
        return self.df


def Fonction_regression():

    # Génération des données
    T, V = datasets.make_regression(n_samples=1000, n_features=6, noise=10)

    #p = 6
    feature_names = [f"X{i}" for i in range(T.shape[1])]

    data_1 = Dataset(X=T, Y=V, features_names=feature_names)

    data_1.add_intercept()       
    df = data_1.to_dataframe()    

    print(df)

    # Régression
    res_regression, intercept, betas = regression(data_1.X, data_1.Y, data_1.features_names)
    Beta = res_regression.coefficients

    print("dataclass contenant les coefficients :", res_regression)

    # Dictionnaire
    dict_coeff = {data_1.features_names[i]: Beta[i] for i in range(len(data_1.features_names))}
    
    print("Dictionnaire des coefficients : ", dict_coeff)

    # Résultats
    res = results(data_1.X, data_1.Y, df, Beta)
    res.predict()
    res.metriques()

    df_extended = res.extend_df()

    print("Extended DataFrame :\n", df_extended.head())
    print("R² :", res.R2)
    print("RMSE :", res.RMSE)


if __name__ == "__main__":
    Fonction_regression()
