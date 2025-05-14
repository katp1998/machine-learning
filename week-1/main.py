import pandas as pd
from sklearn import linear_model
from sklearn.preprocessing import StandardScaler
scale = StandardScaler()

dataset = pd.read_csv (r'cars.csv')

X = dataset[['Weight', 'Volume']]
scaledX = scale.fit_transform(X)

print(scaledX)

scale = StandardScaler()

X = dataset[['Weight', 'Volume']]
y = dataset['CO2']

scaledX = scale.fit_transform(X)
regr = linear_model.LinearRegression()
regr.fit(scaledX, y)
scaled = scale.transform([[2300, 1.3]])


predictedCO2 = regr.predict([scaled[0]])
print(predictedCO2)
