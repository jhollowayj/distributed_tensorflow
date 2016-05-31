try:
   import cPickle as pickle
except:
   import pickle
import numpy as np

# TODO perhaps add in a zip command on the compressed weights to get them even smaller
# If we can do the zipping async, we might find this advantagous. 

def compress_weights(weights):
    ''' Creates a string to send consisting of:
            arraybytes + split token + arrayshape
        for each layer in keras network!
    '''
    weight_string = "" 
    for i, w in enumerate(weights):
        weight_string += w.astype(np.float32).tostring()
        weight_string += "$$SPLIT$$"
        weight_string += pickle.dumps(w.shape)
        # print "=======arrayshape:{}".format(w.shape)
        if i < (len(weights) - 1):
            weight_string += "$$SPLIT$$"
    return weight_string

def decompress_weights(weight_string):
    ''' Inverse of compress_weights above '''
    weight_string = weight_string.split("$$SPLIT$$")
    weights = []
    for i in range(0, len(weight_string), 2):
        arr = np.fromstring(weight_string[i], np.float32)
        shape = pickle.loads(weight_string[i+1])
        # print "=======Org:{}, to new shape:{}".format(arr.shape, shape)
        weights.append(arr.reshape(shape))
    return weights
    


def compress_request(msgType, network_type, network_type_id):
    return pickle.dumps([msgType, network_type, network_type_id])

def decompress_request(msg):
    req = pickle.loads(msg)
    return (req[0], req[1], req[2])


    
def label_compressed_weights(network_type, network_id, compressed_weights):
    return pickle.dumps([network_type, network_id, compressed_weights])

def delabel_compressed_weights(msg):
    obj = pickle.loads(msg)
    return (obj[0], obj[1], obj[2])
    
    
# Yes, I know these are the same as label compressed weights, but that's ok.
# I like having separate names to be more readable in the client and server code.
def compress_msg(msgType, networkType, networkId):
    return pickle.dumps([msgType, networkType, networkId])
def decompress_msg(msg):
    obj = pickle.loads(msg)
    return (obj[0], obj[1], obj[2])