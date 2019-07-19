
# Linear Regression From Scratch With NumPy

Welcome to the first post of the [Implementing Machine Learning Algorithms with Numpy](https://) series in which I'll try to show how one can implement some supervised, unsupervised and semi-supervised algorithms only with numpy package. 

Of course, we will use other useful packages such as `matplotlib`, ` seaborn` and etc. However, the use of other packages may only be limited to data visualization, data manipulation and/or loading datasets (e.g. `sklearn.datasets`) such that we won't take any shortcuts while writing the actual code for machine learning models.

To sum it up, we will be implementing machine learning algorithms from scratch! Isn't that exciting and little bit overwhelming at the same time? Did I mention that it is super fun as well? The first algorithm that we will tackle is linear regression. Since it is the "hello world" algorithm of the machine learning universe, it will be pretty easy to implement it with NumPy. Let's start right away!

## Linear Regression Intuition

Before we write the code for implementation of linear regression, first we need to understand what exactly linear regression is. There are many useful resources out there that makes it quite easy to understand the concept behind regression and particularly linear regression so, I won't be going into much detail here. 

Linear regression is used to make some sense of the data we have at hand by unearthing the relation between target values and features. When we know this relation, we can make predictions about the data that we haven't seen before, in other words, we can infer the target value from feature values. Let's exemplify this: 

Suppose we want to understand how a company X decides what to pay to its employees. There may be so many factors that go into that decision and we go around and ask most of the employees who work there. After a lot of prying and sneaking around, it turns out that some of them earn a lot because they have been working at the company X for quite some time, some of them earn higher than most simply because they get along really well with the boss. Some earn higher because of their qualifications and talent. These three indicators seem to be the major ones. Now, with the information we have gathered, we want to understand the underlying relation between these factors and the salary that is paid to the employees currently. We come up with this oversimpflied equation:

$\textbf{SALARY = } \textrm{(? } \times \textbf{ Qualifications} \textrm{) } + \textrm{(? } \times \textbf{ Length of Service} \textrm{) } + \textrm{(? } \times \textbf{ Buttering up the Boss} \textrm{) }$


We can see from the equation above that the salary is affected by the 3 chosen attributes. These attributes, also called features, affect the salary according to their own weight which is depicted in the equation as question marks simply because we don't actually know what these weights are. 

Now, let's imagine what would happen if we know these weights exactly. Then if we have an employee whose salary we don't know, we can use her features (qualifications, length of service etc.) to predict her salary. That is, we would understand how these features and the target value (salary) are related. 

Turns out, linear regression is used to do exactly that! It is used to get a good estimate of these weights so that they can be used to predict the target value of unseen data. In machine learning literature these weights are often called parameters, hence from now on, we'll adopt that term here as well. 

## Gradient Descent Algorithm

Now that we know "what" linear regression is, we can come to the "how" part. How does this algorithm work? How can we figure out these parameters for linear regression? In machine learning, there is another famous algorithm called "gradient descent" that is used a lot, not only for estimating the parameters for linear regression but for other optimization problems as well. In gradient descent algorithm, parameters of the model (here, that is linear regression) is changed iteratively at each step starting with the initial values of the parameters. 

To remind us once more, parameters (weights) are the numerical values that determine how much each feature affects the target value. We want to know these parameter values exactly, but in real life this is not possible because there may be so many other features (hence, parameters of those features as well) affecting the target value. However, we want them to predict the target value as close as possible to the actual value.  Since, the question marks in the above equation represent the parameter values, we can replace them with initial values like this:

$\textbf{SALARY = } \textrm{(1000 } \times \textbf{ Qualifications} \textrm{) } + \textrm{(200 } \times \textbf{ Length of Service} \textrm{) } + \textrm{(500 } \times \textbf{ Buttering up the Boss} \textrm{) }$

Here, it is obvious that qualifications feature affects salary more than the other features, because its parameter value is higher than the rest. Keep in mind that we have chosen these parameter values intuitively and we will be using them as our initial parameter values, yet these initial values will change at every step of the algorithm towards their optimal values.

Going along with our analogy, suppose we have an initial estimate for the parameters of these features and we went around and asked these questions to the first employee we could find:

 1. For how long have you been working here?
 2. What are your qualifications for your position?
 3. How do you get along with your boss? (Does your boss seem to like or dislike you?) 

For the first question, we told the employee that we would accept an answer in years (1 year, 2 years, 5 years etc.). For the second question, we told the employee that the answer would be any number from 1 to 10 (1 being the least qualified and 10 the most). For the last question, the answer would be a number from -5 to 5. Here, minus represents the negativity of the relationship between the employee and the boss. Therefore, -5 means that the boss quite dislikes the employee, 0 could mean that the boss doesn't even know the employee and/or there is no interaction between the two and +5 means that two of them get along just great. 

When we asked the employee these questions, these are the answers we got:
1. I've have been working here for 10 years.
2. I can honestly say that I'm overqualified for this job. So I would give it a 9.
3. My boss seems to hate me. Whenever I'm around I can see the hatred in his eyes. So I would give it a -4.

Remember that we want to predict the salary based the parameters we have chosen and the answers we've got from the employee. After predicting what the salary would be based on only these answers, we ask the employee what the actual salary is. The difference between the predicted and actual value determines how successful our estimates for these parameters (weights) are. Gradient descent algorithm's job would be to make this difference (predicted - actual) as small as possible. Let's go ahead and call this difference "error", since it represents how much off the actual value is from the predicted value. Now, let's plug the numbers we get from the first employee's answers into our equation:

$\textbf{SALARY = } \textrm{(1000 } \times \textbf{ 10} \textrm{) } + \textrm{(200 } \times \textbf{ 9} \textrm{) } + \textrm{(500 } \times \textbf{ -4} \textrm{) }$

Hence, this shows that our prediction for the salary is:

$\textbf{SALARY}_\textrm{predicted} = 12800 $


Now, we ask the employee what her actual salary is and we calculate the error between the actual and predicted value:

$\textbf{SALARY}_\textrm{actual} = 9800 $

$\textbf{Error } = \textbf{SALARY}_\textrm{predicted} - \textbf{SALARY}_\textrm{actual} = 12800 - 9800 = 3000 $


We see that our error is 3000 which is a lot and we have to make this error as small as possible by tweaking the parameter values appropriately. But how do we do that? How can we decide what is the correct way of changing the parameter values? Obviously, we can make guesses intuitively and change the parameter values (increase or decrease) to make the error small enough. However, this won't be very easy if we have 100 features and not only three. 100 features mean 100 parameter values, remember. Obviously, we have to find a better way than this. Moreover, there is another factor to consider. This error cannot represent only one employee, in other words, we cannot only change the parameter values for one employee since we want this model to be representative for all the employees who work at company X. We have to get the answers from all the employees and plug those numbers into the equation, find the error and change the parameters accordingly. 

Perhaps we could go over all of the employees at company X and sum all of the individual errors, just like this:

$Total Error = \left( \sum_{k=1}^n Error_k \right)^2 $ , where n is the total number of employees who work at company X.

The error function that we have used here (Error = Predicted - Actual) is one of the most basic function to be used in machine learning pipeline, and often it has its certain limitationsso now let's use a more adopted version which is called "ordinary least squares" which is simply the the sum of square differences between the actual and predicted values:

Ordinary Least Squares = 

Now, a quick change of notation is in order: Cost function is often used a lot instead of error. Because, it basically costs us to miss the actual value by a value of predicted-actual. If predicted value was equal to the actual value, the cost would be zero. Therefore, a cost function is a measure of how wrong the model is in terms of its ability to estimate the relationship between feature values and target values. ----> add this to the error part.

After establishing the cost function we can now move on. The whole point of gradient descent algorithm is to minimize the cost function. When we minimize the cost function, we are actually guaranteeing the possible lowest error while increasing the accuracy of our model. We go over our dataset iteratively while updating the parameters at each step. Back to our analogy, remember that we had three parameters (qualifications, length of service, buttering up the boss) that we wanted to change in the direction that minimized the cost function. So checking each data point in our dataset basically means asking each employee who works at company X those 3 questions we convised and using the answers and plugging those numbers into the cost function hence, calculating the cost function and deciding which direction the next step we should take to minimize the cost function.

Now, how do we decide which direction we should go to make the total cost a bit smaller? Calculus comes to our help here. Hence, when we want to minimize a function we take the derivative of the function with respect to a variable and use that derivative to decide which direction to go. In our analogy, the parameters that we have chosen are actually the variables of our cost function because the cost function varies as each parameter varies (variable, duh). We have to take the derivative with respect to each parameter (cough, variable) and update the parameters using those derivative values. In the picture below, we can see the graph of the cost function against just one parameter (Length of Service). Now when we calculate the partial derivative of the cost function with respect to this parameter only, we get the direction we need to move towards for this parameter, in order to reach the local minima whose slope equals to 0.

<img src="img/cost_function.png" width=400 height=200 > <br> <br>

When we take the derivative with respect to each parameter and find the direction we need to move towards, we update each parameter simultaneously:

$
\textit{Length of Service}_\textrm{updated} = \textit{Length of Service}_\textrm{old} - \textbf{ (Learning Rate} \times \textit{Partial Derivative w.r.t. Length of Service)} 
$

This update rule is applied to all of the parameters using their partial derivatives correspondingly. Here, learning rate as it is also called learning step, is the amount that the parameters are updated during learning the optimized parameter values. Learning rate is a configurable hyperparameter, often in the range between 0.0 and 1.0, that controls the rate or speed at which the model learns. If it's high, the model learns quickly, however it's too much high the during the update we might miss the optimal value because we took a really big step. If the learning rate is too low, then the model will take a lot of time to converge to the lowest cost function value. 

So one iteration means asking all of the employees for once (or going over the dataset for once). After one iteration, we update the parameter values accordingly (We'll get to that later). Suppose :

Total cost we get after 1st iteration: 123444
Total cost after 2nd iteration: 88283
Total cost after 3rd iteration: 2234
....
Total cost after 100th iteration: 541


So one iteration means asking all of the employees for once (or going over the dataset for once) and updating the parameter values accordingly. After going through the dataset many times, iterating stops when we reach a point where the cost is low enough for us to decide that we can stop the algorithm and use the parameter values that were updated up until now. Then, we can use those "opitimized" values to predict new target values for new feature values. And those predictions will be pretty good. What do we mean by "optimized" here? Well, now we have found parameter values for our 3 features, so that they can predict new target values with possible lowest error. Hence, we optimized those parameters in our model. This is where learning of the "machine learning" happens indeed. We "learn" the parameters that minimizes our cost function. 


## 1. Importing necessary libraries


```python
import numpy as np
from sklearn.datasets import load_boston
import matplotlib.pyplot as plt
```

First things first, we start by importing necessary libraries to help us along the way. As I have mentioned before, we won't be using any packages that will give us already implemented algorithm models such as `sklearn.linear_model` since it won't help us grasp what is the underlying principles of implementing an algorithm because it is an out-of-the-box (hence, ready-made) solution. We want to do it the hard way, not the easy way.

Moreover, do notice that we can use `sklearn` package (or other packages) to make use of its useful functions, such as loading a dataset, as long as we don't use its already implemented algorithm models.

We will be using:
* `numpy` (obviously) to do all of the vectorized numerical computations on the dataset including the implementation of the algorithm,
* `matplotlib` to plot graphs for better understanding the problem at hand with some visual aid,
*` sklearn.datasets` to load some toy datasets to play around with our written code.


### 2. Loading and exploring the dataset


```python
dataset = load_boston()

X = dataset.data
y = dataset.target[:,np.newaxis]

print("Total samples in our dataset is: {}".format(X.shape[0]))
```

    Total samples in our dataset is: 506


Now, it's time to load the dataset we will be using throughout this post. The `sklearn.datasets` package offers some toy datasets to illustrate the behaviour of some algorithms and we will be using `load_boston()`function to return a regression dataset. Here, `dataset.data` represents the feature samples and `dataset.target` returns the label values. 

It is important to note that, when we are loading the target values, we are adding a new dimension to the data (`dataset.target[:,np.newaxis]`), so that we can use the data as a column vector. Remember, linear algebra makes a distinction between row vectors and column vectors. However, in NumPy there are only n-dimensional arrays and no concept for row and column vectors, per se. We can use arrays of shape `(n, 1)` to imitate column vectors and `(1, n)` for row vectors. Ergo, we can use our target values of shape `(n, )` as a column vector of shape ` (n, 1)` by adding an axis explicitly. Luckily, we can do that with NumPy's own `newaxis` function which is used to increase the dimension of an array by one more dimension, when used once.


```python
def gradient_descent(X, y, theta, alpha, iterations):
    m = len(y)
    J_history = np.zeros((iterations,1))

    for i in range(iterations):
        theta = theta - (alpha/m) * X.T @ (X @ theta - y) 
        J_history[i] = compute_cost(X, y, theta)

    return (J_history, theta)


def compute_cost(X, y, theta):
    m = len(y)
    h = X @ theta
    return (1/(2*m))*np.sum((h-y)**2)



dataset = load_boston()
X = dataset.data
y = dataset.target[:,np.newaxis]

m = len(y)

mu = np.mean(X, 0)
sigma = np.std(X, 0)

X = (X-mu) / sigma

X = np.hstack((np.ones((m,1)),X))
n = np.size(X,1)
theta = np.zeros((n,1))

iterations = 1500
alpha = 0.01

InitialCost = compute_cost(X, y, theta)

print("Initial Cost is: {}".format(InitialCost))

(J_history, theta_optimal) = gradient_descent(X, y, theta, alpha, iterations)

(J_history_2, theta_optimal) = gradient_descent(X, y, theta, 0.001, iterations)

(J_history_3, theta_optimal) = gradient_descent(X, y, theta, 0.1, iterations)

print("Optimal Theta is: ", theta_optimal)

plt.plot(range(len(J_history)), J_history, 'r')
plt.plot(range(len(J_history_2)), J_history_2, 'y')
plt.plot(range(len(J_history_3)), J_history_3, 'g')
plt.title("Convergence Graph of Cost Function")
plt.legend(("alpha: 0.01", "alpha: 0.001", "alpha: 0.1"))
plt.xlabel("Number of Iterations")
plt.ylabel("Cost")
plt.show()
```

    Initial Cost is: 296.0734584980237
    Optimal Theta is:  [[ 2.25328063e+01]
     [-9.28135085e-01]
     [ 1.08154929e+00]
     [ 1.40840005e-01]
     [ 6.81748308e-01]
     [-2.05670784e+00]
     [ 2.67424105e+00]
     [ 1.94568604e-02]
     [-3.10404863e+00]
     [ 2.66206628e+00]
     [-2.07660959e+00]
     [-2.06060107e+00]
     [ 8.49267351e-01]
     [-3.74362129e+00]]



![png](Linear_Regression_files/Linear_Regression_8_1.png)



```python

```
