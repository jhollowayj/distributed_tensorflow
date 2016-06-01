import tensorflow as tf
import numpy as np
from enum import IntEnum

class NetworkType(IntEnum):
    World = 1
    Task = 2
    Agent = 3
    
class Messages(IntEnum):
    NetworkWeightsReady = 1
    RequestingNetworkWeights = 2
    RequestingServerUUID = 3
    # Not sure if there were any other messages to pass, but I figured I'd put this here.

params = {
    'world_input_size': 144,
    'world_to_task': 1000,
    'task_to_agent': 1000,
    'agent_1_action_space_size': 4,
    'agent_2_action_space_size': 4,
}

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
    # elif worldId == 2:  return World2()
    else:
        raise InvalidModelRequest("World ID {} doesn't exist".format(worldId))
def taskBuilder(taskID=1):
    if taskID == 1:    return Task1()
#    elif taskID == 2:  return World2()
    else:
        raise InvalidModelRequest("Task ID {} doesn't exist".format(taskID))
def agentBuilder(agentId=1):
    if agentId == 1:    return Agent1()
#    elif agentId == 2:  return World2()
    else:
        raise InvalidModelRequest("Agent ID {} doesn't exist".format(agentId))
        
#####################################################################################################
#####################################################################################################
class InvalidGradients(Exception):
    def __init__(self, value):
        self.value = value
    def __str__(self):
        return repr(self.value)


class Shared_Model_Object:
    def __init__(self):
        self.model = None
        self.layers = None
        
    def get_model_weights(self):
        ''' Returns list of np.arrays containing the weights '''
        # return [l.eval(self.sess) for l in self.layers]
        return self.sess.run(self.layers)

    def savable_vars(self):
        return self.layers

    def set_model_weights(self, weights):
        ''' Takes in list of np.arrays containing the new weights to use '''
        self._errorCheckIncomingWeights(weights)
        self.sess.run(self.assigns, feed_dict={self.assigns_placeholders[i]:weights[i] for i in range(len(weights))})
                
    def add_gradients(self, weights):
        ''' Takes in list of np.arrays containing gradients to apply (+=) to weights '''
        self._errorCheckIncomingWeights(weights)
        self.sess.run(self.assign_adds, feed_dict={self.assigns_placeholders[i]:weights[i] for i in range(len(weights))})

        # self.sess.run([lay.assign_add(weights[index]) for index, lay in enumerate(self.layers)]) # TODO add placeholders
    
    def _errorCheckIncomingWeights(self, weights):
        ''' Checks to make sure all of the shapes line up, throws error if not the same '''
        if len(weights) != len(self.layers):
            raise InvalidGradients("number of layers didn't match")
        for i in range(len(weights)):
            if weights[i].shape != self.layers[i].eval(self.sess).shape: # yuck!!!!!
                raise InvalidGradients("shape of gradient[{}] didn't match layer shape".format(i))
                
#####################################################################################################
class World1(Shared_Model_Object):
    '''Takes input and expands to 1k nodes.  We can build this up later.
        Input size is standard for worlds, output is standard for worlds
        Input size = 
    '''
    def __init__(self, sess):
        Shared_Model_Object.__init__(self)
        self.sess = sess
        # Layer 1
        self.w1 = tf.Variable(tf.random_normal([params['world_input_size'], params['world_to_task']], stddev=0.01, dtype=tf.float32), name="W1_L1_w")
        self.b1 = tf.Variable(tf.constant(0.1, shape=[params['world_to_task']], dtype=tf.float32), name="W1_L1_b")
        self.assign_w1_placeholder = tf.placeholder(tf.float32, shape=[params['world_input_size'], params['world_to_task']])
        self.assign_b1_placeholder = tf.placeholder(tf.float32, shape=[params['world_to_task']])
        self.assign_w1 = self.w1.assign(self.assign_w1_placeholder)
        self.assign_b1 = self.b1.assign(self.assign_b1_placeholder)
        self.assign_add_w1 = self.w1.assign_add(self.assign_w1_placeholder)
        self.assign_add_b1 = self.b1.assign_add(self.assign_b1_placeholder)
        # outside vars
        self.layers = [ self.w1, self.b1 ]
        self.assigns = [self.assign_w1, self.assign_b1]
        self.assign_adds = [self.assign_add_w1, self.assign_add_b1]
        self.assigns_placeholders = [self.assign_w1_placeholder, self.assign_b1_placeholder]

class World2(Shared_Model_Object):
    '''Takes input and expands to 1k nodes.  We can build this up later.
    Input size is standard for worlds, output is standard for worlds
    Input size = 
    '''
    def __init__(self, sess):
        Shared_Model_Object.__init__(self)
        self.sess = sess
        # Layer 1
        self.w1 = tf.Variable(tf.random_normal([params['world_input_size'], params['world_to_task']], stddev=0.01, dtype=tf.float32), name="W2_L1_w")
        self.b1 = tf.Variable(tf.constant(0.1, shape=[params['world_to_task']], dtype=tf.float32), name="W2_L1_b")
        self.assign_w1_placeholder = tf.placeholder(tf.float32, shape=[params['world_input_size'], params['world_to_task']])
        self.assign_b1_placeholder = tf.placeholder(tf.float32, shape=[params['world_to_task']])
        self.assign_w1 = self.w1.assign(self.assign_w1_placeholder)
        self.assign_b1 = self.b1.assign(self.assign_b1_placeholder)
        self.assign_add_w1 = self.w1.assign_add(self.assign_w1_placeholder)
        self.assign_add_b1 = self.b1.assign_add(self.assign_b1_placeholder)
        # outside vars
        self.layers = [ self.w1, self.b1 ]
        self.assigns = [self.assign_w1, self.assign_b1]
        self.assign_adds = [self.assign_add_w1, self.assign_add_b1]
        self.assigns_placeholders = [self.assign_w1_placeholder, self.assign_b1_placeholder]
        
class World3(Shared_Model_Object):
    '''Takes input and expands to 1k nodes.  We can build this up later.
    Input size is standard for worlds, output is standard for worlds
    Input size = 
    '''
    def __init__(self, sess):
        Shared_Model_Object.__init__(self)
        self.sess = sess
        # Layer 1
        self.w1 = tf.Variable(tf.random_normal([params['world_input_size'], params['world_to_task']], stddev=0.01, dtype=tf.float32), name="W3_L1_w")
        self.b1 = tf.Variable(tf.constant(0.1, shape=[params['world_to_task']], dtype=tf.float32), name="W3_L1_b")
        self.assign_w1_placeholder = tf.placeholder(tf.float32, shape=[params['world_input_size'], params['world_to_task']])
        self.assign_b1_placeholder = tf.placeholder(tf.float32, shape=[params['world_to_task']])
        self.assign_w1 = self.w1.assign(self.assign_w1_placeholder)
        self.assign_b1 = self.b1.assign(self.assign_b1_placeholder)
        self.assign_add_w1 = self.w1.assign_add(self.assign_w1_placeholder)
        self.assign_add_b1 = self.b1.assign_add(self.assign_b1_placeholder)
        # outside vars
        self.layers = [ self.w1, self.b1 ]
        self.assigns = [self.assign_w1, self.assign_b1]
        self.assign_adds = [self.assign_add_w1, self.assign_add_b1]
        self.assigns_placeholders = [self.assign_w1_placeholder, self.assign_b1_placeholder]

#####################################################################################################
class Task1(Shared_Model_Object):
    def __init__(self, sess):
        Shared_Model_Object.__init__(self)
        self.sess = sess
        # Layer 1
        self.w1 = tf.Variable(tf.random_normal([params['world_to_task'], params['task_to_agent']], stddev=0.01, dtype=tf.float32), name="T1_L1_w")
        self.b1 = tf.Variable(tf.constant(0.1, shape=[params['task_to_agent']], dtype=tf.float32), name="T1_L1_b")
        self.assign_w1_placeholder = tf.placeholder(tf.float32, shape=[params['world_to_task'], params['task_to_agent']])
        self.assign_b1_placeholder = tf.placeholder(tf.float32, shape=[params['task_to_agent']])
        self.assign_w1 = self.w1.assign(self.assign_w1_placeholder)
        self.assign_b1 = self.b1.assign(self.assign_b1_placeholder)
        self.assign_add_w1 = self.w1.assign_add(self.assign_w1_placeholder)
        self.assign_add_b1 = self.b1.assign_add(self.assign_b1_placeholder)
        # outside vars
        self.layers = [ self.w1, self.b1 ]
        self.assigns = [self.assign_w1, self.assign_b1]
        self.assign_adds = [self.assign_add_w1, self.assign_add_b1]
        self.assigns_placeholders = [self.assign_w1_placeholder, self.assign_b1_placeholder]
class Task2(Shared_Model_Object):
    def __init__(self, sess):
        Shared_Model_Object.__init__(self)
        self.sess = sess
        # Layer 1
        self.w1 = tf.Variable(tf.random_normal([params['world_to_task'], params['task_to_agent']], stddev=0.01, dtype=tf.float32), name="T2_L1_w")
        self.b1 = tf.Variable(tf.constant(0.1, shape=[params['task_to_agent']], dtype=tf.float32), name="T2_L1_b")
        self.assign_w1_placeholder = tf.placeholder(tf.float32, shape=[params['world_to_task'], params['task_to_agent']])
        self.assign_b1_placeholder = tf.placeholder(tf.float32, shape=[params['task_to_agent']])
        self.assign_w1 = self.w1.assign(self.assign_w1_placeholder)
        self.assign_b1 = self.b1.assign(self.assign_b1_placeholder)
        self.assign_add_w1 = self.w1.assign_add(self.assign_w1_placeholder)
        self.assign_add_b1 = self.b1.assign_add(self.assign_b1_placeholder)
        # outside vars
        self.layers = [ self.w1, self.b1 ]
        self.assigns = [self.assign_w1, self.assign_b1]
        self.assign_adds = [self.assign_add_w1, self.assign_add_b1]
        self.assigns_placeholders = [self.assign_w1_placeholder, self.assign_b1_placeholder]
class Task3(Shared_Model_Object):
    def __init__(self, sess):
        Shared_Model_Object.__init__(self)
        self.sess = sess
        # Layer 1
        self.w1 = tf.Variable(tf.random_normal([params['world_to_task'], params['task_to_agent']], stddev=0.01, dtype=tf.float32), name="T3_L1_w")
        self.b1 = tf.Variable(tf.constant(0.1, shape=[params['task_to_agent']], dtype=tf.float32), name="T3_L1_b")
        self.assign_w1_placeholder = tf.placeholder(tf.float32, shape=[params['world_to_task'], params['task_to_agent']])
        self.assign_b1_placeholder = tf.placeholder(tf.float32, shape=[params['task_to_agent']])
        self.assign_w1 = self.w1.assign(self.assign_w1_placeholder)
        self.assign_b1 = self.b1.assign(self.assign_b1_placeholder)
        self.assign_add_w1 = self.w1.assign_add(self.assign_w1_placeholder)
        self.assign_add_b1 = self.b1.assign_add(self.assign_b1_placeholder)
        # outside vars
        self.layers = [ self.w1, self.b1 ]
        self.assigns = [self.assign_w1, self.assign_b1]
        self.assign_adds = [self.assign_add_w1, self.assign_add_b1]
        self.assigns_placeholders = [self.assign_w1_placeholder, self.assign_b1_placeholder]

#####################################################################################################
class Agent1(Shared_Model_Object):
    def __init__(self, sess):
        Shared_Model_Object.__init__(self)
        self.sess = sess
        # Layer 1
        self.w1 = tf.Variable(tf.random_normal([params['task_to_agent'], params['agent_1_action_space_size']], stddev=0.01, dtype=tf.float32), name="A1_L1_w")
        self.b1 = tf.Variable(tf.constant(0.1, shape=[params['agent_1_action_space_size']], dtype=tf.float32), name="A1_L1_b")
        #assigning stuff
        self.assign_w1_placeholder = tf.placeholder(tf.float32, shape=[params['task_to_agent'], params['agent_1_action_space_size']])
        self.assign_b1_placeholder = tf.placeholder(tf.float32, shape=[params['agent_1_action_space_size']])
        self.assign_w1 = self.w1.assign(self.assign_w1_placeholder)
        self.assign_b1 = self.b1.assign(self.assign_b1_placeholder)
        self.assign_add_w1 = self.w1.assign_add(self.assign_w1_placeholder)
        self.assign_add_b1 = self.b1.assign_add(self.assign_b1_placeholder)
        # outside vars
        self.layers = [ self.w1, self.b1 ]
        self.assigns = [self.assign_w1, self.assign_b1]
        self.assign_adds = [self.assign_add_w1, self.assign_add_b1]
        self.assigns_placeholders = [self.assign_w1_placeholder, self.assign_b1_placeholder]
class Agent2(Shared_Model_Object):
    def __init__(self, sess):
        Shared_Model_Object.__init__(self)
        self.sess = sess
        # Layer 1
        self.w1 = tf.Variable(tf.random_normal([params['task_to_agent'], params['agent_2_action_space_size']], stddev=0.01, dtype=tf.float32), name="A2_L1_w")
        self.b1 = tf.Variable(tf.constant(0.1, shape=[params['agent_2_action_space_size']], dtype=tf.float32), name="A2_L1_b")
        #assigning stuff
        self.assign_w1_placeholder = tf.placeholder(tf.float32, shape=[params['task_to_agent'], params['agent_2_action_space_size']])
        self.assign_b1_placeholder = tf.placeholder(tf.float32, shape=[params['agent_2_action_space_size']])
        self.assign_w1 = self.w1.assign(self.assign_w1_placeholder)
        self.assign_b1 = self.b1.assign(self.assign_b1_placeholder)
        self.assign_add_w1 = self.w1.assign_add(self.assign_w1_placeholder)
        self.assign_add_b1 = self.b1.assign_add(self.assign_b1_placeholder)
        # outside vars
        self.layers = [ self.w1, self.b1 ]
        self.assigns = [self.assign_w1, self.assign_b1]
        self.assign_adds = [self.assign_add_w1, self.assign_add_b1]
        self.assigns_placeholders = [self.assign_w1_placeholder, self.assign_b1_placeholder]
class Agent3(Shared_Model_Object):
    def __init__(self, sess):
        Shared_Model_Object.__init__(self)
        self.sess = sess
        # Layer 1
        self.w1 = tf.Variable(tf.random_normal([params['task_to_agent'], params['agent_2_action_space_size']], stddev=0.01, dtype=tf.float32), name="A3_L1_w")
        self.b1 = tf.Variable(tf.constant(0.1, shape=[params['agent_2_action_space_size']], dtype=tf.float32), name="A3_L1_b")
        #assigning stuff
        self.assign_w1_placeholder = tf.placeholder(tf.float32, shape=[params['task_to_agent'], params['agent_2_action_space_size']])
        self.assign_b1_placeholder = tf.placeholder(tf.float32, shape=[params['agent_2_action_space_size']])
        self.assign_w1 = self.w1.assign(self.assign_w1_placeholder)
        self.assign_b1 = self.b1.assign(self.assign_b1_placeholder)
        self.assign_add_w1 = self.w1.assign_add(self.assign_w1_placeholder)
        self.assign_add_b1 = self.b1.assign_add(self.assign_b1_placeholder)
        # outside vars
        self.layers = [ self.w1, self.b1 ]
        self.assigns = [self.assign_w1, self.assign_b1]
        self.assign_adds = [self.assign_add_w1, self.assign_add_b1]
        self.assigns_placeholders = [self.assign_w1_placeholder, self.assign_b1_placeholder]
#####################################################################################################

debugging = False
if debugging:
    sess = tf.Session()
    print sess
    w = World1(sess)
    sess.run(tf.initialize_all_variables())

    # print type(w.layers)
    # print len(w.layers)
    # print type(w.layers[0].eval(w.sess))
    # print len(w.layers[0].eval(w.sess))
    # print w.layers[0].eval(w.sess)
    # print np.array(w.layers[0].value())
    
    ### Test 1: get and set weights works ###
    x = w.get_model_weights()
    start = x[0][0][0] # Store the first guy
    print x[0][0]
    w.set_model_weights([np.zeros(layer.shape) for layer in w.get_model_weights()])
    print w.get_model_weights()[0][0]
    assert w.get_model_weights()[0][0][0] == 0
    w.add_gradients(x)
    assert start == w.get_model_weights()[0][0][0]
    print w.get_model_weights()[0][0]
    ### Test 1: get and set weights works ###
