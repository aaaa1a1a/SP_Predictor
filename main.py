import modelling as md
import download_data as dd
import pandas as pd
from sklearn.model_selection import train_test_split

data = pd.read_csv(dd.file_name("data"))
data = data.ffill(axis=0)
data = data.drop(["Date"], axis=1)

xTrain, xTest, yTrain, yTest = train_test_split(data.drop(["Signal"], axis=1),
                                                data["Signal"], test_size=0.2,
                                                random_state=42)

ada = md.fit_ada_boost(xTrain, yTrain)
prediction = pd.DataFrame()
results = pd.DataFrame()
results["true_y"] = yTest
results["prediction"] = ada.predict(xTest)
results.to_csv("results.csv")
print(results.shape)
print((results["true_y"] == results["prediction"]).value_counts())
