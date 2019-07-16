from sklearn.datasets import load_boston, fetch_california_housing
import linear_regression as lr
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

dataset = fetch_california_housing()

X = dataset.data
y = dataset.target

X_train, X_test, y_train, y_test = train_test_split(\
                X, y, test_size=0.3, random_state=42)

regressor = lr.LinearRegression(X_train, y_train).fit()

train_accuracy = regressor.score()
test_accuracy = regressor.score(X_test, y_test)

predictions = regressor.predict(X_test)

params = regressor.get_params()
intercept = regressor.intercept_
coef = regressor.coef_

print("Training accuracy is: {}".format(train_accuracy))
print("Test accuracy is: {}".format(test_accuracy))
#print(params)
#print(regressor.intercept_)
#print(regressor.coef_)

reg = LinearRegression().fit(X_train, y_train)
pred = reg.predict(X_test)
print(reg.score(X_test, y_test))  # Train accuracy for sklearn's model

plt.scatter(y_test, predictions)
plt.scatter(y_test, pred)
