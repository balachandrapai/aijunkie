---
layout: post
title: "Mathematics of Machine Learning"
date: 2018-12-21
categories: ["mathematicalFoundations", Introduction]
---

Machine learning is all about creating an algorithm that can learn from the data to make a prediction. The big 4 math disciplines that make up machine learning are linear algebra, probability theory, calculus, and statistics. Machine learning is built on these mathematical prerequisites.<br/>
* Statistics is at the core of everything.
* Calculus is used to learn and optimize the model.
* Linear algebra makes it feasible to run the algorithm on massive datasets.
* Probability helps predict the likelihood of an event occuring.

In mathematics, the field of statistics contains a collection of techniques that extract useful information from data. It's a tool to create an understanding from the set of numbers. Statistical inference is a process of making a prediction about a larger population of data based on a smaller sample.<br/>
Consider the following points x and y where the relationship between them is a straight line $y = mx + c$.<br/>

```python
import numpy as np
x = np.array([-1.0, 0.0, 1.0, 2.0, 3.0, 4.0], dtype=float)
y = np.array([-3.0, -1.0, 1.0, 3.0, 5.0, 7.0], dtype=float)
```
```python
import matplotlib.pyplot as plt
plt.scatter(x,y)
plt.plot(x,y)
plt.show()
```
<img src="{{ "/pictures/linear.png" | prepend: site.baseurl }}" align="middle" width="50%" height="50%"/>
**Linear regression** is a statistical inference technique used to study the relationship between two variables x and y. Variable x is regarded as an independent variable. The other variable y is regarded as the dependent variable. The way we can represent linear regression is by using the equation $y = mx + c$.<br/>
* y is the prediction.
* x is the input.
* c is the point where the line intercepts the y axis.
* m is the slope of the line.<br/>
We try to learn the value of m and c such that when a line is drawn using these values will be the best fit for our dataset. An error function is used to measure how good the predicted values are. **Least square error** function shown below is one such statistical loss function.<br/>
$\sum_{i=1}^n (Y_{t} - \hat Y_{t})^2$
* $Y_{t}$ - is the actual value.
* $\hat Y_{t}$ - is the predicted value. <br/>

Below is the graph of x,y (with different possible values of m and c) and z (potential error values for every combination of m and c). The bottom of the bowl is the ideal value of m and c for which error value is the least. That is the line of best fit.
This is computed using calculus - The study of change.<br/>

<img src="{{ "/pictures/gradient-descent.jpg" | prepend: site.baseurl }}" align="middle" width="50%" height="50%"/>
If there multiple variable to consider then it is a **multivariate regression**. **Linear algebra** is the branch of math that deals with study of multi variate spaces and the linear transformation between them.<br/>
$Y_{t} = a_{1}+b_{2}X_{2t}+b_{3}X_{3t}+e_{t}$
* $Y_{t}$ - dependent variable.
* $a_{1}$ - intercept.
* $b_{2}$, $b_{3}$ - constant (partial regression coefficient).
* $X_{2}$, $X_{3}$ - explanatory variable.
* $e_{t}$ - error term.<br/>

Probability is the measure of likelihood of an outcome. **Logistic regression** is one such probablistic technique. Contrary to linear model where the value is predicted, logistic regression is used to predict the probability of an occurence. Since the probability ranges between 0 and 100, a sigmoid function is employed.<br/> 

An example for linear fitting.<br/>
```python
import keras
from keras.models import Sequential
from keras.layers import Dense

model = Sequential([Dense(1, input_shape=[1])])
model.compile(optimizer='adam', loss='mean_squared_error')
```

```python
model.fit(x,y, epochs=500)
```

```python
model.predict([10.0])
```
    array([[17.375637]], dtype=float32)
