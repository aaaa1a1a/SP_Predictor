import modelling as md
import download_data as dd
import pandas as pd
import feature_engineering as fe
from sklearn.model_selection import train_test_split

'''
data_file = dd.file_name("data")
data = pd.read_csv(data_file)
data = data.ffill(axis=0)

for col in data.columns:
    if col != "Date" and col != "Volume" and col != "Signal":
        data[col] = fe.log_returns(data, col)
        data[col] = data[col].replace(to_replace=0, method='ffill')

data.to_csv(dd.file_name("data_normalized"))
'''

data_file = dd.file_name("data_normalized")
data = pd.read_csv(data_file)

data = data.drop(["Date"], axis=1)

xTrain, xTest, yTrain, yTest = train_test_split(data.drop(["Signal"], axis=1),
                                                data["Signal"], test_size=0.2,
                                                random_state=42)

prediction = pd.DataFrame()
results = pd.DataFrame()
results["true_y"] = yTest

"""
ada = md.fit_ada_boost(xTrain, yTrain)
results["prediction"] = ada.predict(xTest)
print("Adaboost Classifier")
print(md.get_results(results["true_y"], results["prediction"]))

rf = md.fit_random_forest(xTrain, yTrain)
results["prediction"] = rf.predict(xTest)
print("Random Forest Classifier")
print(md.get_results(results["true_y"], results["prediction"]))
"""

knn = md.fit_knn(xTrain, yTrain)
results["prediction"] = knn.predict(xTest)
print("KNN")
print(md.get_results(results["true_y"], results["prediction"]))
