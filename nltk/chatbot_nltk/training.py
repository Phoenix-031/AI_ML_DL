import json
import numpy as np
import tensorflow as tf
import tflearn
import random
import pickle

import nltk
from nltk.stem.lancaster import LancasterStemmer
stemmer = LancasterStemmer()

with open('intents.json') as file:
    data = json.load(file)

try:
    with open("data.pickle", "rb") as f:
        words, classes, training, output = pickle.load(f)

except:
    words = []
    classes = []
    docs = []
    docs_x = []
    ignore = ['?','.','!','/','"',"'"]

    for intent in data['intents']:
        for pattern in intent['patterns']:
            wrd = nltk.word_tokenize(pattern)
            words.extend(wrd)
            docs.append(wrd)
            docs_x.append(intent['tag'])

        if intent['tag'] not in classes:
            classes.append(intent['tag'])

    words = [stemmer.stem(w.lower()) for w in words if w not in ignore]
    words = sorted(list(set(words)))

    training = []
    output = []

    output_empty = [0 for _ in range(len(classes))]


    for i,doc in enumerate(docs):
        bag = []
        
        wrds = [stemmer.stem(w) for w in doc]

        for w in words:
            if w in wrds:
                bag.append(1)
            else:
                bag.append(0)

        output_row = output_empty[:]
        output_row[classes.index(docs_x[i])] = 1

        training.append(bag)
        output.append(output_row)


    training = np.array(training)
    output = np.array(output)

    with open("data.pickle", "wb") as f:
         pickle.dump((words, classes, training, output),f)


# tf.reset_default_graph()

net = tflearn.input_data(shape=[None, len(training[0])])
net = tflearn.fully_connected(net,8)
net = tflearn.fully_connected(net,8)
net = tflearn.fully_connected(net,len(output[0]),activation = "softmax")
net = tflearn.regression(net)

model = tflearn.DNN(net)

# try:
#     model.load("model.tflearn")

# except:
model.fit(training,output,n_epoch = 5000,batch_size = 8,show_metric = True)
model.save("model.tflearn")



def bag_words(s, words):
    bag = [0 for _ in range(len(words))]

    s_words = nltk.word_tokenize(s)
    s_words = [stemmer.stem(word.lower()) for word in s_words]

    for val in s_words:
        for i,w in enumerate(words):
            if w == val:
                bag[i] = 1
            

    return np.array(bag)

def chat():
    print("online")
    while 1:
        inp = input("You:")
        if inp.lower() == "quit":
            break

        res = model.predict([bag_words(inp,words)])[0]
        res_ind = np.argmax(res)
        tag = classes[res_ind]
        
        if res[res_ind] > 0.7:
            for tg in data['intents']:
                if tg['tag'] == tag:
                    response = tg['responses']

            print(random.choice(response))
        else:
            for tg in data['intents']:
                if tg['tag'] == "noanswer":
                    response = tg['responses']

            print(random.choice(response))
            
            
# if __name__ == '__main__':
#     chat()




