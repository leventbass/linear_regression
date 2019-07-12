from sklearn.datasets import load_boston
from linear_regression import LinearRegression
from sklearn.model_selection import train_test_split

dataset = load_boston()

X = dataset.data
y = dataset.target

X_train, X_test, y_train, y_test = train_test_split(\
                X, y, test_size=0.33, random_state=42)

regressor = LinearRegression(X_train, y_train).fit()

train_accuracy = regressor.score()
test_accuracy = regressor.score(X_test, y_test)

print("Training accuracy is: {}".format(train_accuracy))
print("Test accuracy is: {}".format(test_accuracy))
