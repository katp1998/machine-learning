import pandas as pd
import numpy 
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
#Import libraries

x = numpy.random.normal(3, 1, 100)
y = numpy.random.normal(150, 40, 100) / x
#Add data to x and y from the dataset - numpy.random.normal - this adds some jitter (extra values to improve to results)

plt.xlabel("Number of minutes before making a purchase")
plt.ylabel("Money spent")
#Add labels to the axis

train_x = x[:80]
train_y = y[:80]
#Spilt the data into 80%

test_x = x[80:]
test_y = y[80:]
#Split the data into 20%

mypolymodel = numpy.poly1d(numpy.polyfit(train_x, train_y, 4))
#method that lets us make a polynomial mode (Number 4 is the degree - change this between 1 - 10 to see the changes)

polyline = numpy.linspace(0, 6)
#Then specify how the line will display, we start at position 1, and end at position 6:

r2 = r2_score(train_y, mypolymodel(train_x))
r2testdata = r2_score(test_y, mypolymodel(test_x))
#Calculate the r-squared values

plt.scatter(x, y)
plt.scatter(train_x, train_y)
plt.plot(polyline, mypolymodel(polyline))
plt.show()
#Plot the diagrams

print(r2)
print(r2testdata)
print(mypolymodel(22.54))
#print outputs to the console

