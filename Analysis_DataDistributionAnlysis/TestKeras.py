#coding=utf-8
import tensorflow as tf
import numpy as np
np.random.seed(1337)
import keras
from keras.models import Sequential
from keras.layers import Dense
import matplotlib.pyplot as plt

# create some data
X = np.linspace(-1, 1, 200)
np.random.shuffle(X)
Y = 0.5*X + 2 + np.random.normal(0, 0.05, (200,))

#plot data
plt.scatter(X, Y)
plt.show()

X_train, Y_train = X[:160], Y[:160]
X_test, Y_test = X[160:], Y[160:]
plt.scatter(X_test, Y_test)
plt.show()

# build a neural network from the 1st layer to the last layer
model = Sequential()
model.add(Dense(units=1, input_shape=(1,)))

# choose loss function and optimizing method
model.compile(optimizer='sgd', loss='mse')

#training
print('\ntraining')
for step in range(301):
    cost=model.train_on_batch(X_train, Y_train)
    if (step %100 == 0):
        print('cost ', cost)

#test
print('\n testing')
cost = model.evaluate(X_test, Y_test, batch_size=400)
print('test cost ', cost)
W, b = model.layers[0].get_weights()
print('weights=', W, '\nbias=', b)

# plotting the prediction
print('\n predicting')
Y_pred = model.predict(X_test)
plt.scatter(X_test, Y_test) # scatter 点
plt.plot(X_test, Y_pred)  # plot 线
plt.show()
print('\n prediction')


