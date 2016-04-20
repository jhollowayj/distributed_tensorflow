# -*- coding: utf-8 -*-
"""
Created on Mon Mar 21 10:50:15 2016

@author: tetchart
"""

from NNpractice import *



# Import the MNIST data
#import cPickle as pickle
import gzip

f = gzip.open('mnist.pkl.gz', 'rb')
train_set, valid_set, test_set = pickle.load(f)
f.close()

train_dataset = train_set[0]
train_labels = train_set[1]
valid_dataset = valid_set[0]
valid_labels = valid_set[1]
test_dataset = test_set[0]
test_labels = test_set[1]
#pickle_file = 'notMNIST.pickle'

"""with open(pickle_file, 'rb') as f:
    save = pickle.load(f)
    train_dataset = save['train_dataset']
    train_labels = save['train_labels']
    valid_dataset = save['valid_dataset']
    valid_labels = save['valid_labels']
    test_dataset = save['test_dataset']
    test_labels = save['test_labels']
    del save  # hint to help gc free up memory"""
 
image_size = 28
num_labels = 10

import numpy as np
def reformat(dataset, labels):
    dataset = dataset.reshape((-1, image_size * image_size)).astype(np.float32)
    # Map 0 to [1.0, 0.0, 0.0 ...], 1 to [0.0, 1.0, 0.0 ...]
    labels = (np.arange(num_labels) == labels[:,None]).astype(np.float32)
    return dataset, labels
train_dataset, train_labels = reformat(train_dataset, train_labels)
valid_dataset, valid_labels = reformat(valid_dataset, valid_labels)
test_dataset, test_labels = reformat(test_dataset, test_labels)









# Server
w1 = world1()
w2 = world2()
a1 = agent1()


# Client
model = Sequential()
model.add(model_from_json(w1.getJsonModel()))
model.add(model_from_json(a1.getJsonModel()))

model.compile(loss='categorical_crossentropy', optimizer=SGD(lr=0.01, momentum=0.9, nesterov=True))
model.fit(train_dataset, train_labels, nb_epoch=1, batch_size=100)

weights = model.get_weights()
w1W = weights[0:4]
w1Pic = pickle.dumps(weights[0:4])
a1Pic = pickle.dumps(weights[4:])

w1UnPic = pickle.loads(w1Pic)

w1.updatePicWeights(w1Pic)
a1.updatePicWeights(a1Pic)

# USE CALLBACKS TO GET DELTA WEIGHTS

model1 = Sequential()
model1.add(model_from_json(w1.getJsonModel()))
model1.add(model_from_json(a1.getJsonModel()))
model1.compile(loss='categorical_crossentropy', optimizer=SGD(lr=0.01, momentum=0.9, nesterov=True))
  
W1weights1 = pickle.loads(w1.getPicWeights())
A1weights1 = pickle.loads(a1.getPicWeights())
weights1 = W1weights1 + A1weights1

weights2 = model1.get_weights()


model1.set_weights(weights2)
model1.fit(train_dataset, train_labels, nb_epoch=1, batch_size=32)












