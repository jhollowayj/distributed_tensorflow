from keras.models import * #Sequential
from keras.optimizers import * #SGD
from keras.layers.core import * #Dense, Activation, Flatten, Reshape
from keras.layers.convolutional import * #Convolution2D
from pprint import pprint
#####################################################################################################
#####################################################################################################
class Shared_Model_Object:
    def __init__(self):
        self.model = None

    def get_model(self):
        return self.model
        
    def get_model_weights(self):
        return self.model.get_weights()
        
    def get_json_model(self):
        return self.model.to_json()
        
    def update_weights(self, weights):
        self.model.set_weights(weights)
        
    def update_pic_weights(self, picWeights):
        weights = pickle.loads(picWeights)
        self.model.set_weights(weights)
        
    def get_pic_weights(self):
        weights = self.model.get_weights()
        return pickle.dumps(weights)

#####################################################################################################
class World1(Shared_Model_Object):
    def __init__(self):
        Shared_Model_Object.__init__(self)
        
        dense1 = Dense(output_dim=64, input_dim=784, init="glorot_uniform", activation="relu")
        dense2 = Dense(output_dim=32, input_dim=64, init="glorot_uniform", activation="relu")
        
        self.model = Sequential()
        self.model.add(dense1)
        self.model.add(dense2)     

#####################################################################################################
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
