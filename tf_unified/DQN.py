import numpy as np
import tensorflow as tf
import time
class DQN:
    def __init__(self, input_dims = 2, num_act = 4,
            eps = 1.0, discount = 0.95, lr =  0.0002,
            rms_eps = 1e-6, rms_decay=0.99,
            allow_local_nn_weight_updates=False,
            requested_gpu_vram_percent=0.01,
            device_to_use=0, verbose = 0):
        self.params = {
            'input_dims': input_dims,
            'layer_1_hidden': 1000,
            'layer_2_hidden': 1000,
            'num_act': num_act,
            'discount': discount,
            'lr': lr,
            'rms_eps': rms_eps,
            'rms_decay': rms_decay,
            'allow_local_nn_weight_updates': allow_local_nn_weight_updates,
            'requested_gpu_vram_percent': requested_gpu_vram_percent, 
            'verbose': verbose,
        }
        
        if self.params['allow_local_nn_weight_updates']:
            print "NOTE: network is allowed to update his own weights!" 

        device2use = {-1: "/cpu:0", 0: "/gpu:0", 1: "/gpu:1"}[device_to_use]
        print "using Device: {}".format(device2use)
        with tf.device(device2use):
            if device_to_use == -1:
                self.sess = tf.Session()
            else:
                gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=self.params['requested_gpu_vram_percent'],
                                            allow_growth=True, deferred_deletion_bytes=1)
                self.sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
        
            self.x = tf.placeholder(tf.float32,[None, self.params['input_dims']], name="nn_x")
            self.q_t = tf.placeholder(tf.float32,[None], name="nn_q_t")

            self.actions = tf.placeholder(tf.float32, [None, self.params['num_act']], name="nn_actions")
            self.rewards = tf.placeholder(tf.float32, [None], name="nn_rewards")
            self.terminals = tf.placeholder(tf.float32, [None], name="nn_terminals")

            # GENERIC OPTIMIZER VARIABLES #  
            self.discount = tf.constant(self.params['discount'], name="nn_discount")
            self.yj = tf.add(self.rewards, tf.mul(1.0-self.terminals, tf.mul(self.discount, self.q_t)), name="nn_yj_whatever-that-is")

            num_types_per_layer = [1,3,3] # [W,T,A]
            num_layers = len(num_types_per_layer) # 3
            ### NETWORK WEIGHTS ###
            dims = [self.params['input_dims'], self.params['layer_1_hidden'], self.params['layer_2_hidden'], self.params['num_act']]
            self.layers = {}
            for l in range(num_layers): # World, Agent, Task
                self.layers[l] = {}
                for i in range(num_types_per_layer[l]): # W:{1},  A:{1,2,3},  T:{1,2,3}
                    self.layers[l][i] = (
                        tf.Variable(tf.random_normal([dims[l], dims[l+1]], stddev=0.01, dtype=tf.float32), name="nn_L{}_{}_w".format(l, i)),
                        tf.Variable(tf.constant(0.1,    shape=[dims[l+1]]),             dtype=tf.float32,  name="nn_L{}_{}_b".format(l, i)))
            ### NETOWRKS ###
            self.o1,self.o2,self.y,self.Q_pred,self.cost, self.rmsprop_min = {}, {}, {}, {}, {}, {}
            for w in range(num_types_per_layer[0]):
                self.o1[w], self.o2[w], self.y[w], self.Q_pred[w], self.cost[w], self.rmsprop_min[w] = {}, {}, {}, {}, {}, {}
                for t in range(num_types_per_layer[1]):
                    self.o1[w][t], self.o2[w][t], self.y[w][t], self.Q_pred[w][t], self.cost[w][t], self.rmsprop_min[w][t] = {}, {}, {}, {}, {}, {} 
                    for a in range(num_types_per_layer[2]):
                        # NETWORK WEIGHTS
                        w0, b0 = self.layers[0][w]
                        w1, b1 = self.layers[1][w]
                        w2, b2 = self.layers[2][w]
                        # WORLD LAYER
                        self.o1[w][t][a] = tf.nn.relu(tf.add(tf.matmul(self.x,w0),b0), name="nn_L1_output")
                        # TASK LAYER
                        self.o2[w][t][a] = tf.nn.relu(tf.add(tf.matmul(self.o1[w][t][a],w1),b1), name="nn_L2_output")
                        # AGENT LAYER
                        self.y[w][t][a]  = tf.add(tf.matmul(self.o2[w][t][a],w2),b2, name="y_OR_nn_L3_output")
                        
                        # OPTIMIZER VARIABLES
                        self.Q_pred[w][t][a] = tf.reduce_sum(tf.mul(self.y[w][t][a],self.actions), reduction_indices=1, name="nn_q_pred")
                        self.cost[w][t][a]   = tf.reduce_sum(tf.pow(tf.sub(self.yj, self.Q_pred[w][t][a]), 2), name="nn_cost")
                        self.rmsprop_min[w][t][a] = tf.train.RMSPropOptimizer(self.params['lr'],self.params['rms_decay'],0.0,self.params['rms_eps']).minimize(self.cost[w][t][a])

        self.sess.run(tf.initialize_all_variables())
        print "###\n### Networks initialized\n### Ready to begin\n###"
    
    def train(self, W,T,A, bat_s,bat_a,bat_t,bat_n,bat_r):
        '''Please note: This allows the networks to change their weights!'''
        feed_dict={self.x: bat_s, self.q_t: np.zeros(bat_n.shape[0]), self.actions: bat_a, self.terminals:bat_t, self.rewards: bat_r}
        q_t = self.sess.run(self.y[W][T][A],feed_dict=feed_dict)
        q_t = np.amax(q_t,axis=1)
        feed_dict={self.x: bat_s, self.q_t: q_t, self.actions: bat_a, self.terminals:bat_t, self.rewards: bat_r}
        
        _, costs = self.sess.run([self.rmsprop_min[W][T][A], self.cost[W][T][A]], feed_dict=feed_dict)
        
        return costs

    def q(self, W,T,A, bat_s):
        return self.sess.run(self.y[W][T][A], feed_dict = {
            self.x: bat_s,
            self.q_t: np.zeros(1),
            self.actions: np.zeros((1, self.params['num_act'])),
            self.terminals:np.zeros(1),
            self.rewards: np.zeros(1)
        })
        
    def save_weights(self, name, boolean_for_something):
        # w1, b2 = self.sess.run([self.w1, self.b2])
        return
