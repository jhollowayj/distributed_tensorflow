# -*- coding: utf-8 -*-
"""
Created on Tue Apr 19 18:06:37 2016

@author: tetchart
"""

from keras.models import * #Sequential
from keras.optimizers import * #SGD
from keras.layers.core import * #Dense, Activation, Flatten, Reshape
from keras.layers.convolutional import * #Convolution2D
import cPickle as pickle


class pracNN:
    
    def __init__(self):
        sec1_dense1 = Dense(output_dim=64, input_dim=784, init="glorot_uniform", activation="relu")
        sec1_dense2 = Dense(output_dim=32, input_dim=64, init="glorot_uniform", activation="relu")
        
        self.sec1_model = Sequential()
        self.sec1_model.add(sec1_dense1)
        self.sec1_model.add(sec1_dense2)
        
        ##############################################################################################        
        
        sec2_dense1 = Dense(output_dim=128, input_dim=784, init="glorot_uniform", activation="relu")
        sec2_dense2 = Dense(output_dim=64, input_dim=128, init="glorot_uniform", activation="relu")
        sec2_dense3 = Dense(output_dim=32, input_dim=64, init="glorot_uniform", activation="relu")
        
        self.sec2_model = Sequential()
        self.sec2_model.add(sec2_dense1)
        self.sec2_model.add(sec2_dense2)
        self.sec2_model.add(sec2_dense3)

        ##############################################################################################

        self.out1_softplus = Dense(output_dim=10, input_dim=32, init="glorot_uniform", activation="softplus")
        self.out2_softplus = Dense(output_dim=10, input_dim=32, init="glorot_uniform", activation="softplus")

    def getModel(self, whichModel):

        if (whichModel[0] == "sec1"):
            if (whichModel[1] == "out1"):
                self.sec1_model.add(self.out1_softplus)
                return self.sec1_model
            elif (whichModel[1] == "out2"):
                self.sec1_model.add(self.out2_softplus)
                return self.sec1_model
            else: print "Sorry, that's not a valid model..."
            
        if (whichModel[0] == "sec2"):
            if (whichModel[1] == "out1"):
                self.sec2_model.add(self.out1_softplus)
                return self.sec2_model
            elif (whichModel[1] == "out2"):
                self.sec2_model.add(self.out2_softplus)
                return self.sec2_model
            else: print "Sorry, that's not a valid model..."
            
            
#####################################################################################################
#####################################################################################################
class shared_model_object:
    def __init__(self):
        self.model = None

    def getModel(self):
        return self.model
        
    def getJsonModel(self):
        return self.model.to_json()
        
    def updateWeights(self, weights):
        self.model.set_weights(weights)
        
    def updatePicWeights(self, picWeights):
        weights = pickle.loads(picWeights)
        self.model.set_weights(weights)
        
    def getPicWeights(self):
        weights = self.model.get_weights()
        return pickle.dumps(weights)

#####################################################################################################
class world1(shared_model_object):
    def __init__(self):
        dense1 = Dense(output_dim=64, input_dim=784, init="glorot_uniform", activation="relu")
        dense2 = Dense(output_dim=32, input_dim=64, init="glorot_uniform", activation="relu")
        
        self.model = Sequential()
        self.model.add(dense1)
        self.model.add(dense2)     

#####################################################################################################
class world2(shared_model_object):
    def __init__(self):
        dense1 = Dense(output_dim=128, input_dim=784, init="glorot_uniform", activation="relu")
        dense2 = Dense(output_dim=64, input_dim=128, init="glorot_uniform", activation="relu")
        dense3 = Dense(output_dim=32, input_dim=64, init="glorot_uniform", activation="relu")
        
        self.model = Sequential()
        self.model.add(dense1)
        self.model.add(dense2)
        self.model.add(dense3)

#####################################################################################################
class agent1(shared_model_object):
    def __init__(self):
        out1_softplus = Dense(output_dim=10, input_dim=32, init="glorot_uniform", activation="softplus")
        
        self.model = Sequential()
        self.model.add(out1_softplus)

