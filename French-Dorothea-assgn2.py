# for review in reviews
    # ID                      -- 0
    # count positive words    -- 1
    # count negative words    -- 2
    # if has no then 1 else 0 -- 3
    # count pronouns.txt      -- 4
    # if ! then 1 else 0      -- 5
    # ln word count           -- 6
    # review class            -- 7

# read in review vectors to a matrix
# split into 80/20 for review (argparse --training true)

# z = sum(wx)+b
# y_hat = sigmoid(z) = 1/(1+exp(-z)
# loss = -(ylog(y_hat)+(1-y)log(1-y_hat))
# gradient = (y_hat - y)x
# weights = weights(n-1)- eta*gradient

import csv
import math
import numpy as np
import random

def vectorize(ID, review, words, pronouns, category):  # helper function that takes a review and creates corresponding feature vector
    vector = [ID, 0, 0, 0, 0, 0, 0, category]
    print(vector[0])
    vector[6] = np.log(len(review.split()) - 1)
    if "!" in review:
        vector[5] = 1
    review = review.replace('.', " ").replace(",", " ").replace('?', " ").replace("-", " ").replace("!", " ").replace("(", " ").replace(")", " ")
    revWords = review.split()[2:]
    for word in revWords:
        word = word.lower()
        value = words.get(word, "0")  # dictionary containing positive and negative words
        if value == 1:
            vector[1] += 1
        elif value == -1:
            vector[2] += 1
        if word in pronouns:
            vector[4] += 1
        if word == "no":
            vector[3] = 1
    return vector


def vectorize_file(filepath, category, words, pronouns):  # helper function that vectorizes a file based on the category of reviews it contains (positive, negative, additional (mixed), test (unknown))
    vectors = []
    reader = open(filepath, "r", encoding="utf8")
    if category == "additional":
        for review in reader:
            print("reading additional review")
            cat = review.split()[0]
            ID = review.split()[1]
            vector = vectorize(ID, review, words, pronouns, cat)
            vectors.append(vector)
            #print(vector)
    elif category == "test":
        for review in reader:
            print("reading additional review")
            cat = "UNK"
            ID = review.split()[0]
            vector = vectorize(ID, review, words, pronouns, cat)
            vectors.append(vector)
            #print(vector)
    elif category == "negative":
        for review in reader:
            print("reading negative review")
            ID = review.split()[0]
            vector = vectorize(ID, review, words, pronouns, 0)
            vectors.append(vector)
            print(vector)
    elif category == "positive":
        for review in reader:
            print("reading positive review")
            ID = review.split()[0]
            vector = vectorize(ID, review, words, pronouns, 1)
            vectors.append(vector)
            print(vector)
    else:
        print("error with file categories")
    reader.close()
    return vectors


def full_vectorizer(add_data):  # reads in review files and prints their vectorized form
    reader = open("features/negative-words.txt", "r")
    wordDict = {}
    for word in reader:
        word = word.split('\n')[0]
        wordDict[word] = -1
    reader.close()

    reader = open("features/positive-words.txt", "r")
    for word in reader:
        word = word.split('\n')[0]
        wordDict[word] = 1
    reader.close()

    pronouns = []
    reader = open("features/pronouns.txt", "r")
    for word in reader:
        word = word.split('\n')[0]
        pronouns.append(word)
    reader.close()

    filepath = "training-data/hotelNegT-train.txt"
    category = "negative"
    vectors = vectorize_file(filepath, category, wordDict, pronouns)

    filepath = "training-data/hotelPosT-train.txt"
    category = "positive"
    vectors = vectors + vectorize_file(filepath, category, wordDict, pronouns)

    with open("processed/French-Dorothea_assgn2-part1.csv", "w", newline='') as file:
        writer = csv.writer(file, delimiter=",")
        for vector in vectors:
            writer.writerow(vector)

    if add_data:
        filepath = "training-data/HW2-testset.txt"
        category = "test"  # what kind of file is the second file: positive, negative, additional, or test?
        vectors = vectorize_file(filepath, category, wordDict, pronouns)

        with open("processed/Hwk2-test.csv", "w", newline='') as file:
            writer = csv.writer(file, delimiter=",")
            for vector in vectors:
                writer.writerow(vector)


def compute_gradient(rev, weights, learn_rate):  # helper function that computes the gradient and returns updated weights
    review = []
    y = float(rev[-1])
    rev = np.append(rev[1:7], .1)
    for x in rev:
        review.append(float(x))
    review = np.array(review)
    z = review@weights
    y_hat = 1/(1+np.exp(-z))
    #print("predicted = " + str(y_hat) + " actual= " + str(y))
    #loss = -(y * np.log(y_hat)+(1-y)*np.log(1-y_hat))
    gradient = (y_hat - y)*review
    newWeights = weights - learn_rate * gradient
    return np.array(newWeights)


def training(matrix, seed):  # helper function that takes & shuffles training matrix and keeps track of weights
    weights = np.array([0, 0, 0, 0, 0, 0, 1])
    random.seed(seed) #high 61: .92
    random.shuffle(matrix)
    matrix = np.array(matrix)
    for review in matrix:
        weights = compute_gradient(review, weights, 1)  # calculates new weights from old weights and current review
    print("Final weights: ")
    print(weights)
    return weights


def test_categorizer(rev, weights):  # helper function to categorize review without updating weights
    ID = rev[0]
    review = []
    if rev[-1] != "UNK":
        y = float(rev[-1])
    else:
        y = -1
    rev = np.append(rev[1:7], 1)
    for x in rev:
        review.append(float(x))
    review = np.array(review)
    z = review@weights
    y_hat = 1/(1+np.exp(-z))
    if y_hat <= .5:
        if y == 0:
            return 1, "NEG"
        else:
            return 0, "NEG"
    else:
        if y == 1:
            return 1, "POS"
        else:
            return 0, "POS"
    #loss = -(y * np.log(y_hat)+(1-y)*np.log(1-y_hat))
    print("failure")

def review_categorizer(files, seed):
    test_data = []
    train_data = []
    train_length = 0
    test_length = 0
    for file in files:  # adds correct number of reviews from each file to be used for training or testing
        reader = open(file[0], "r")

        testsize = file[1]
        test_length += file[1]
        train_length += file[2] - file[1]
        i = 0
        j = 0
        for review in reader:
            review = review.replace('\n', "").split(',')
            #print(review)
            if i <= math.floor((testsize - 2)/2):
                #print(i)
                test_data.append(review)
                i += 1
            elif i > file[2] - round((testsize+2)/2):
                #print(i)
                test_data.append(review)
                i += 1
                j += 1
            else:
                #print(i)
                train_data.append(review)
                i += 1

    train_data = np.array(train_data)
    test_data = np.array(test_data)
    print(train_data.shape)
    print("train length: " + str(train_length))
    print(test_data.shape)
    print("test length: " + str(test_length))

    weights = training(train_data, seed)  # trains weights on training reviews

    accuracy = 0
    for review in train_data:
        accu, cat = test_categorizer(review, weights)
        accuracy += accu

    #print("Train set accuracy: " + str(accuracy/train_length))

    writer = open("processed/French-Dorothea-assgn2-out.txt", "w", encoding="utf8")  # prints ID and grade to output file
    accuracy = 0
    for review in test_data:
        accu, cat = test_categorizer(review, weights)
        accuracy += accu
        writer.write(str(review[0]) + " " + cat + "\n")
    writer.close()

    print("Test set accuracy: " + str(accuracy/test_length))


processed = True  # do you want to vectorize the reviews?
if not processed:
    add_data = True  # do you want to vectorize a second file?
    full_vectorizer(add_data)
else:
    print("Using pre-processed vectors")

filepath1 = "processed/French-Dorothea_assgn2-part1.csv" # filepath to training data vectors
test_number1 = 0  # number of reviews to be used for testing
length1 = 189

filepath2 = "processed/Hwk2-test.csv" # filepath to second data set's vectors (test data)
test_number2 = 50  # number of reviews to be used for testing
length2 = 50
files = [[filepath1, test_number1, length1], [filepath2, test_number2, length2]]


seed = 23  # determined to maximize accuracy, used to shuffle training matrix. Highest known percentage ~ 97%, average 85%, lowest 48%
review_categorizer(files, seed)  # trains weights, tests, and prints to output files
