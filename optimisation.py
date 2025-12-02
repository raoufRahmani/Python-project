from b import Dataset, regression, results
import statsmodels.api as sm
import numpy as np
import pandas as pd
from sklearn import datasets
# Partie 1


def fonction_fix(p):
    T, V = datasets.make_regression(n_samples=1000, n_features=6, noise=10)
    feature_list = [f"X{i}" for i in range(p)]
    data_1 = Dataset(X=T, Y=V, features_names=feature_list)
    return  data_1


def test_add_intercept():
    data_1 = fonction_fix(6)
    data_1.add_intercept()
    assert data_1.X.shape[0] == data_1.n ,f"The number of samples should remain {data_1.n} after adding intercept."
    assert data_1.X.shape[1] == data_1.p + 1 ,f"The number of features should be {data_1.p + 1}. after adding intercept."
    assert data_1.features_names[0] == "constante","The first column name should be 'constante'."
    assert len(data_1.features_names) == data_1.p + 1, "The Feature names list length is incorrect."


def test_to_dataframe():
    data_1 = fonction_fix(6)
    data_1.add_intercept()
    df = data_1.transform_to_dataframe()
    assert type(df) == pd.DataFrame, "The output should be a pandas DataFrame."
    assert df.shape[0] == data_1.n, f"The number of rows in DataFrame should be {data_1.n}."
    assert df.shape[1] == data_1.p + 2, f"The number of columns in DataFrame should be {data_1.p + 2}."
    assert df.columns[0] == "constante", "The first column name should be 'constante'."
    assert df.columns[-1] == "variable_expliquée", "The last column name should be 'variable_expliquée'."


def test_regression():
    X=np.array([[1, 1], [1, 2], [2, 2], [2, 3]])
    Y=np.array([1, 2, 2, 3])
    feature_names=["constante", "X1"]
    reg_model= regression(X, Y, feature_names)
    sm_model = sm.OLS(Y, X).fit()
    assert reg_model.coefficients.shape[0] == X.shape[1], "The number of coefficients should match the number of features"
    assert np.allclose(reg_model.coefficients, sm_model.params), "The coefficients are not computed correctly."


def test_predict():
    X=np.array([[1, 1], [1, 2], [2, 2], [2, 3]])
    Y=np.array([1, 2, 2, 3])
    feature_names=["constante", "X1"]
    reg_model= regression(X, Y, feature_names)
    Beta = reg_model.coefficients
    Y_pred = np.dot(X, Beta)
    sm_model = sm.OLS(Y, X).fit()
    sm_Y_pred = sm_model.predict(X)
    assert np.allclose(Y_pred, sm_Y_pred), "The predicted values are not computed correctly."

def test_metriques():
    X=np.array([[1, 1], [1, 2], [2, 2], [2, 3]])
    Y=np.array([1, 2, 2, 3])
    feature_names=["constante", "X1"]
    reg_model= regression(X, Y, feature_names)
    Beta = reg_model.coefficients
    reg_model = results(X, Y, None, Beta)
    reg_model.predict()
    reg_model.metriques()

    local_R2 = reg_model.R2
    local_RMSE = reg_model.RMSE

    sm_model = sm.OLS(Y, X).fit()
    R2 = sm_model.rsquared
    RMSE = np.sqrt(np.mean((Y - sm_model.predict(X))**2))

    assert np.allclose(R2, local_R2), "Les deux R2 doivent être égaux"
    assert np.allclose(RMSE, local_RMSE), "Les deux RMSE doivent être égaux"

def test_extend_df():
    X=np.array([[1, 1], [1, 2], [2, 2], [2, 3]])
    Y=np.array([1, 2, 2, 3])
    feature_names=["constante", "X1"]
    reg_model= regression(X, Y, feature_names)
    df = pd.DataFrame(X, columns=feature_names)
    df["variable_expliquée"] = Y
    res = results(X, Y, df, reg_model.coefficients)
    res.predict()
    extended_df = res.extend_df()
    assert "valeurs_prédites" in extended_df.columns, "The new DataFrame should contain the column of the predicted values"
    assert "erreurs" in extended_df.columns, "The DataFrame should contain errors column."
    assert extended_df.shape[0] == X.shape[0], "The number of rows in the extended DataFrame should match the number of initial samples."