#for review is reviews
    #ID                      -- 0
    #count positive words    -- 1
    #count negative words    -- 2
    #if has no then 1 else 0 -- 3
    #count pronouns.txt      -- 4
    #if ! then 1 else 0      -- 5
    #ln word count           -- 6
    #review class            -- 7
import csv

import numpy as np


def vectorize(review, words, pronouns, category):
    vector = ["", 0,0,0,0,0,0,category]
    vector[0] = review[0:7]
    #print(review.split())
    vector[6] = np.log(len(review.split()) - 1)
    if "!" in review:
        vector[5] = 1
    review = review.replace('.', " ").replace(",", " ").replace('?', " ").replace("-", " ").replace("!", " ").replace("(", " ").replace(")", " ")
    revWords = review.split()[2:]
    #print(revWords)
    for word in revWords:
        word = word.lower()
        #print(word)
        value = words.get(word, "0")
        if value == 1:
            #print(word + " -- positive")
            vector[1] += 1
        elif value == -1:
            #print(" -- negative")
            vector[2] += 1
        if word in pronouns:
            vector[4] += 1
        if word == "no":
            vector[3] = 1
    return vector



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

#print(wordDict)
#print(pronouns)

vectors = []
reader = open("training-data/hotelNegT-train.txt", "r", encoding="utf8")
for review in reader:
    #review = word.split('\n')[0]
    print("reading negative review")
    vector = vectorize(review, wordDict, pronouns, 0)
    vectors.append(vector)
    print(vector)
reader.close()

reader = open("training-data/hotelPosT-train.txt", "r", encoding="utf8")
for review in reader:
    #review = word.split('\n')[0]
    print("reading positive review")
    vector = vectorize(review, wordDict, pronouns, 1)
    vectors.append(vector)
    #print(vector)
reader.close()

with open("processed/reviews.csv", "w", newline='') as file:
    writer = csv.writer(file, delimiter=",")
    for vector in vectors:
        writer.writerow(vector)

