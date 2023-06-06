import pandas as pd
import ssl
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split, KFold, cross_val_score

ssl._create_default_https_context = ssl._create_unverified_context
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
headers = ["sepal_length", "sepal_width", "petal_length", "petal_width",
           "class"]

df = pd.read_csv(url, names=headers)
print(df)
#Podział na zbiory

X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=2023)

#Inicjalizacja modelu

nb = GaussianNB()

#Trenowanie modelu

nb.fit(X_train,y_train)

#Predykcja i ocena modelu

kfold = KFold(n_splits=5,random_state=2023,shuffle=True)
scores = cross_val_score(nb, X_train, y_train, cv=kfold, scoring="accuracy")

print("Wyniki sprawdzianu krzyżowego:")
print(scores)
print(f"Średnia dokładność: {scores.mean()}")

