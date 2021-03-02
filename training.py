#read in review vectors to a matrix
#split into 80/20 for review (argparse --training true)


# z = sum(wx)+b
# y_hat = sigmoid(z) = 1/(1+exp(-z)
# loss = -(ylog(y_hat)+(1-y)log(1-y_hat))
# gradient = (y_hat - y)x
# weights = weights(n-1)- eta*gradient

#weights = [0,0,0,0,0,0,1]
import random

import numpy as np
from decimal import Decimal

def compute_gradient(rev, weights, learn_rate):
    #print(rev)
    newWeights = [0, 0, 0, 0, 0, 0, 0]
    review = []
    y = float(rev[-1])
    #print(y)
    rev = np.append(rev[1:7], .1)
    for x in rev:
        review.append(float(x))
    #print(review)  # pos, neg, no, pron., !, len, bias
    review = np.array(review)
    z = review@weights
    y_hat = 1/(1+np.exp(-z))
    print(y_hat)
    #print("predicted = " + str(y_hat) + " actual= " + str(y))
    loss = -(y * np.log(y_hat)+(1-y)*np.log(1-y_hat))
    #print("loss: " + str(loss))
    #print(weights)
    for index, w in enumerate(weights):
        gradient = (y_hat - y)*review[index]
        newWeights[index] = weights[index] - learn_rate * gradient
    #print(weights)
    #print(newWeights)
    return np.array(newWeights)

def test_categorizer(rev, weights):
    review = []
    y = float(rev[-1])
    ID = rev[0]
    rev = np.append(rev[1:7], 1)
    for x in rev:
        review.append(float(x))
    #print(review)  # pos, neg, no, pron., !, len, bias
    review = np.array(review)
    z = review@weights
    y_hat = 1/(1+np.exp(-z))
    #print(ID + " " + str(y))
    if y_hat <= .5:
        #print("Predicted: 0 ")
        if y == 0:
            return 1
    else:
        #print("Predicted: 1 ")
        if y == 1:
            return 1
    #loss = -(y * np.log(y_hat)+(1-y)*np.log(1-y_hat))
    #print("failure")
    return 0

reader = open("processed/reviews.csv", "r")

withheld = []
matrix = []

testsize = 19  # Set this variable to determine number of positive (testsize) and negative (testsize - 1) exmaples to be withheld as a test set

i = 0
for review in reader:
    review = review.replace('\n', "").split(',')
    #print(review)
    if i <= testsize - 2:
        withheld.append(review)
        i += 1
    elif i >= 189 - testsize:
        withheld.append(review)
        i += 1
    else:
        matrix.append(review)
        i += 1

weights = np.array([0, 0, 0, 0, 0, 0, 1])

# pick random review while some aren't used
weights = compute_gradient(matrix[0], weights, 1)

random.seed(12)
random.shuffle(matrix)
matrix = np.array(matrix)
print(matrix.shape)

for review in matrix:
    weights = compute_gradient(review, weights, 1)

random.shuffle(withheld)
withheld = np.array(withheld)
print(withheld.shape)

accuracy = 0
for review in withheld:
    accuracy += test_categorizer(review, weights)


print(accuracy/37)
print(weights)
accuracy = 0
for review in matrix:
    accuracy += test_categorizer(review, weights)

print(accuracy/152)

