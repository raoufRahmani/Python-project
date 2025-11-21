from dataclasses import dataclass
import numpy as np
from sklearn import datasets


#Partie 1
@dataclass
class Dataset:
    X: np.ndarray
    Y: np.ndarray
    features_names: list
    n: int
    p: int


def add_intercept(vecteur_1: np.ndarray, K: np.ndarray):
    M = np.column_stack([vecteur_1, K])
    return M


def to_dataframe(A: np.ndarray, B: np.ndarray, noms: list):
    df = pd.DataFrame(A, columns=noms)
    df["variable_expliquée"] = B
    return df


T, V = datasets.make_regression(n_samples=1000, n_features=6, noise=10)

p = 6
feature_list = [f"X{i}" for i in range(p)]
feature_list.insert(0, "constante")

data_1 = Dataset(X=T, Y=V, features_names=feature_list, n=1000, p=6)

vecteur = np.ones((data_1.n, 1))
X_avec_constante = add_intercept(vecteur_1=vecteur, K=data_1.X)

df = to_dataframe(X_avec_constante, data_1.Y, feature_list)
print(df)


# Regression
@dataclass
class regression_lineaire:
    coefficients: np.ndarray
    features_names: list


def regression(data) -> regression_lineaire:
    Xt_X = np.dot(X_avec_constante.T, X_avec_constante)
    Xt_Y = np.dot(X_avec_constante.T, data_1.Y)

    Beta = np.dot(np.linalg.inv(Xt_X), Xt_Y)
    intercept = Beta[0]
    betas = Beta[1:]
    return Beta, intercept, betas


resultats_regression = regression(data_1)
Beta, intercept, betas = resultats_regression
# remplire la dataclass

res_regression = regression_lineaire(coefficients=Beta, features_names=feature_list)
print("dataclass qui contient le vecteur des coefficients et les noms des variables explicatives ;", res_regression)

# Dictionnaire
dict_coeff = {feature_list[i]: Beta[i] for i in range(len((feature_list)))}
print("Dictionnare contenant les coefficients : ", dict_coeff)

# 3eme partie

import pandas as pd


class results:
    def __init__(self, X, Y, df):
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
        SCT = np.sum((self.Y - ybar) ** 2) / len(self.Y)
        SCR = np.sum(self.erreurs ** 2) / len(self.Y)
        self.R2 = (SCT - SCR) / SCT
        self.RMSE = np.sqrt(SCR)

    def extend_df(self):
        self.df["valeurs_prédites"] = self.valeurs_prédites
        self.df["erreurs"] = self.erreurs
        return self.df


# Create un objet pour extraire les éléments de la class
res = results(X_avec_constante, data_1.Y, df)

# predictions
res.predict()

#
res.metriques()

# Extend the dataframe
df_extended = res.extend_df()

print("Extended DataFrame : \n", df_extended.head())
print("R² :", res.R2)
print("RMSE :", res.RMSE)
