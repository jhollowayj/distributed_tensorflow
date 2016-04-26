from keras.models import * #Sequential
from keras.optimizers import * #SGD
from keras.layers.core import * #Dense, Activation, Flatten, Reshape
from keras.layers.convolutional import * #Convolution2D

from enum import Enum


class NetworkType(Enum):
    World = 1
    Task = 2
    Agent = 3
    
class Messages(Enum):
    NetworkWeightsReady = 1
    RequestingNetworkWeights = 2
    # Not sure if there were any other messages to pass, but I figured I'd put this here.

#####################################################################################################
class InvalidModelRequest(Exception):
    def __init__(self, value):
        self.value = value
    def __str__(self):
        return repr(self.value)

def builder(worldId = 1, taskId = 1, agentId = 1):
    return (worldBuilder(worldId), \
            taskBuilder(taskId), \
            agentBuilder(agentId))
        
def worldBuilder(worldId=1):
    if worldId == 1:    return World1()
    elif worldId == 2:  return World2()
    else:
        raise InvalidModelRequest("World ID {} doesn't exist".format(worldId))
def taskBuilder(taskID=1):
#    if taskID == 1:    return World1()
#    elif taskID == 2:  return World2()
#    else:
    raise InvalidModelRequest("Task ID {} doesn't exist".format(taskID))
def agentBuilder(agentId=1):
    if agentId == 1:    return World1()
#    elif agentId == 2:  return World2()
    else:
        raise InvalidModelRequest("Agent ID {} doesn't exist".format(agentId))
        
        
#####################################################################################################
#####################################################################################################
class Shared_Model_Object:
    def __init__(self):
        self.model = None

    def get_model(self):
        return self.model
        
    def get_model_weights(self):
        return self.model.get_weights()
        
    def set_model_weights(self, weights):
        self.model.set_weights(weights)

    def get_json_model(self):
        return self.model.to_json()
        
#####################################################################################################
class World1(Shared_Model_Object):
    def __init__(self):
        Shared_Model_Object.__init__(self)
        
        dense1 = Dense(output_dim=64, input_dim=784, init="glorot_uniform", activation="relu")
        dense2 = Dense(output_dim=32, input_dim=64, init="glorot_uniform", activation="relu")
        
        self.model = Sequential()
        self.model.add(dense1)
        self.model.add(dense2)     

class World2(Shared_Model_Object):
    def __init__(self):
        Shared_Model_Object.__init__(self)

        dense1 = Dense(output_dim=128, input_dim=784, init="glorot_uniform", activation="relu")
        dense2 = Dense(output_dim=64, input_dim=128, init="glorot_uniform", activation="relu")
        dense3 = Dense(output_dim=32, input_dim=64, init="glorot_uniform", activation="relu")
        
        self.model = Sequential()
        self.model.add(dense1)
        self.model.add(dense2)
        self.model.add(dense3)

#####################################################################################################
class Agent1(Shared_Model_Object):
    def __init__(self):
        Shared_Model_Object.__init__(self)

        out1_softplus = Dense(output_dim=10, input_dim=32, init="glorot_uniform", activation="softplus")
        
        self.model = Sequential()
        self.model.add(out1_softplus)


#################################################################
#             vvvvvvvvvvv Junk these vvvvvvvvvvv
#################################################################
class World3(Shared_Model_Object):
    def __init__(self):
        Shared_Model_Object.__init__(self)

        out1_softplus = Dense(output_dim=10, input_dim=32, init="glorot_uniform", activation="softplus")
        
        self.model = Sequential()
        self.model.add(out1_softplus)

class Agent2(Shared_Model_Object):
    def __init__(self):
        Shared_Model_Object.__init__(self)

        out1_softplus = Dense(output_dim=10, input_dim=32, init="glorot_uniform", activation="softplus")
        
        self.model = Sequential()
        self.model.add(out1_softplus)
class Agent3(Shared_Model_Object):
    def __init__(self):
        Shared_Model_Object.__init__(self)

        out1_softplus = Dense(output_dim=10, input_dim=32, init="glorot_uniform", activation="softplus")
        
        self.model = Sequential()
        self.model.add(out1_softplus)
        
class Task1(Shared_Model_Object):
    def __init__(self):
        Shared_Model_Object.__init__(self)

        out1_softplus = Dense(output_dim=10, input_dim=32, init="glorot_uniform", activation="softplus")
        
        self.model = Sequential()
        self.model.add(out1_softplus)

class Task2(Shared_Model_Object):
    def __init__(self):
        Shared_Model_Object.__init__(self)

        out1_softplus = Dense(output_dim=10, input_dim=32, init="glorot_uniform", activation="softplus")
        
        self.model = Sequential()
        self.model.add(out1_softplus)
class Task3(Shared_Model_Object):
    def __init__(self):
        Shared_Model_Object.__init__(self)

        out1_softplus = Dense(output_dim=10, input_dim=32, init="glorot_uniform", activation="softplus")
        
        self.model = Sequential()
        self.model.add(out1_softplus)



