import json
import numpy as np

# TODO perhaps add in a zip command on the compressed weights to get them even smaller
# If we can do the zipping async, we might find this advantagous. 

def compress_weights(weights):
    '''  '''
    weight_string = "" 
    for i, w in enumerate(weights):
        weight_string += w.tostring()
        if i < (len(weights) - 1):
            weight_string += "$$SPLIT$$"
    return weight_string

def decompress_weights(weight_string):
    weight_string = weight_string.split("$$SPLIT$$")
    weights = []
    for ws in weight_string:
        weights.append(np.fromstring(ws))
    return weights
    


def compress_request(msgType, network_type, network_type_id):
    return json.dumps([msgType, network_type, network_type_id])

def decompress_request(msg):
    req = json.load(msg)
    return (req[0], req[1], req[2])


    
def label_compressed_weights(network_type, network_id, compressed_weights):
    return json.dumps(network_type, network_id, compressed_weights)

def delabel_compressed_weights(msg):
    obj = json.loads(msg)
    return (obj[0], obj[1], obj[2])
    
    
# Yes, I know these are the same as label compressed weights, but that's ok.
# I like having separate names to be more readable in the client and server code.
def compress_msg(msgType, networkType, networkId):
    return json.dumps([msgType, networkType, networkId])
def decompress_msg(msg):
    obj = json.loads(msg)
    return (obj[0], obj[1], obj[2])