
def Fonction_regression():

    T, V = datasets.make_regression(n_samples=1000, n_features=6, noise=100)


    feature_names = [f"X{i}" for i in range(T.shape[1])]

    data_1 = Dataset(X=T, Y=V, features_names=feature_names)

    data_1.add_intercept()       
    df = data_1.transform_to_dataframe()    

    print(df)

    # Régression
    res_regression = regression(data_1.X, data_1.Y, data_1.features_names)
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



Fonction_regression()


# deuxieme version
# Réorganiser la fonction pricipale en sous fonctions 

def generer_donnees():
    T, V = datasets.make_regression(n_samples=1000, n_features=6, noise=100)
    feature_names = [f"X{i}" for i in range(T.shape[1])]
    return T, V, feature_names


def preparer_dataset(T, V, feature_names):
    data_1 = Dataset(X=T, Y=V, features_names=feature_names)
    data_1.add_intercept()
    df = data_1.transform_to_dataframe()
    return data_1, df


def faire_regression(data_1):
    res_regression = regression(data_1.X, data_1.Y, data_1.features_names)
    Beta = res_regression.coefficients
    return res_regression, Beta


def coeff_en_dict(data_1, Beta):
    dict_coeff = {data_1.features_names[i]: Beta[i] for i in range(len(data_1.features_names))}
    return dict_coeff


def analyser_resultats(data_1, df, Beta):
    res = results(data_1.X, data_1.Y, df, Beta)
    res.predict()
    res.metriques()
    df_extended = res.extend_df()
    return res, df_extended


def Fonction_regression():

    # Partie données
    T, V, feature_names = generer_donnees()

    # Dataset
    data_1, df = preparer_dataset(T, V, feature_names)
    print(df)

    # Régression
    res_regression, Beta = faire_regression(data_1)
    print("dataclass contenant les coefficients :", res_regression)

    # Dictionnaire
    dict_coeff = coeff_en_dict(data_1, Beta)
    print("Dictionnaire des coefficients : ", dict_coeff)

    # Résultats + métriques
    res, df_extended = analyser_resultats(data_1, df, Beta)

    print("Extended DataFrame :\n", df_extended.head())
    print("R² :", res.R2)
    print("RMSE :", res.RMSE)

Fonction_regression()
