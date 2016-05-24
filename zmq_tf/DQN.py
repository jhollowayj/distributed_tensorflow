import numpy as np
import tensorflow as tf
import time

import RProp

class DQN:
    def __init__(self, input_dims = 2, num_act = 4,
            eps = 1.0, discount = 0.90, lr =  0.0002,
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
            'learning_rate_start':  0.003,
            'learning_rate_end':    0.003, #0.000001,
            'learning_rate_decay':  3000,
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

            ### Network ###
            # Layer 1
            layer_1_hidden = self.params['layer_1_hidden']
            self.w1 = tf.Variable(tf.random_normal([self.params['input_dims'], layer_1_hidden], stddev=0.01, dtype=tf.float32), name="nn_L1_w")
            self.b1 = tf.Variable(tf.constant(0.1, shape=[layer_1_hidden]), dtype=tf.float32, name="nn_L1_b")
            self.ip1 = tf.add(tf.matmul(self.x,self.w1),self.b1, name="nn_L1_ip")
            self.o1 = tf.nn.relu(self.ip1, name="nn_L1_output")
            # Layer 2        
            layer_2_hidden = self.params['layer_2_hidden']
            self.w2 = tf.Variable(tf.random_normal([layer_1_hidden, layer_2_hidden], stddev=0.01, dtype=tf.float32), name="nn_L2_w")
            self.b2 = tf.Variable(tf.constant(0.1, shape=[layer_2_hidden]), dtype=tf.float32, name="nn_L2_b")
            self.ip2 = tf.add(tf.matmul(self.o1,self.w2),self.b2, name="nn_L2_ip")
            self.o2 = tf.nn.relu(self.ip2, name="nn_L2_output")
            # Last layer
            self.w3 = tf.Variable(tf.random_normal([layer_2_hidden, self.params['num_act']], stddev=0.01, dtype=tf.float32), name="nn_L3_w")
            self.b3 = tf.Variable(tf.constant(0.1, shape=[self.params['num_act']], dtype=tf.float32), name="nn_L3_b")
            self.y = tf.add(tf.matmul(self.o2,self.w3),self.b3, name="y_OR_nn_L3_output")

            ##
            self.all_layers = [self.w1, self.b1, self.w2, self.b2, self.w3, self.b3]
            ##

            ### Gradients ###
            self.clear_gradients()
            self.assign_w1_placeholder = tf.placeholder(tf.float32, shape=[self.params['input_dims'], layer_1_hidden])
            self.assign_b1_placeholder = tf.placeholder(tf.float32, shape=[layer_1_hidden])
            self.assign_w2_placeholder = tf.placeholder(tf.float32, shape=[layer_1_hidden, layer_2_hidden])
            self.assign_b2_placeholder = tf.placeholder(tf.float32, shape=[layer_2_hidden])
            self.assign_w3_placeholder = tf.placeholder(tf.float32, shape=[layer_2_hidden, self.params['num_act']])
            self.assign_b3_placeholder = tf.placeholder(tf.float32, shape=[self.params['num_act']]) 
            
            self.assign_w1 = self.w1.assign(self.assign_w1_placeholder)
            self.assign_b1 = self.b1.assign(self.assign_b1_placeholder)
            self.assign_w2 = self.w2.assign(self.assign_w2_placeholder)
            self.assign_b2 = self.b2.assign(self.assign_b2_placeholder)
            self.assign_w3 = self.w3.assign(self.assign_w3_placeholder)
            self.assign_b3 = self.b3.assign(self.assign_b3_placeholder)
            
            self.assign_placeholders = [self.assign_w1_placeholder, self.assign_b1_placeholder, self.assign_w2_placeholder, self.assign_b2_placeholder, self.assign_w3_placeholder, self.assign_b3_placeholder]
            self.assign_new_weights = [self.assign_w1, self.assign_b1, self.assign_w2, self.assign_b2, self.assign_w3, self.assign_b3]
            self.compgrad_vars = [ self.w1, self.b2, self.w2, self.b2, self.w3, self.b3 ]
            ### end Gradients ###
            
            #Q,Cost,Optimizer
            self.discount = tf.constant(self.params['discount'], name="nn_discount")
            self.yj = tf.add(self.rewards, tf.mul(1.0-self.terminals, tf.mul(self.discount, self.q_t)), name="nn_yj_whatever-that-is")
            self.Q_pred = tf.reduce_sum(tf.mul(self.y,self.actions), reduction_indices=1, name="nn_q_pred")
            self.cost = tf.reduce_sum(tf.pow(tf.sub(self.yj, self.Q_pred), 2), name="nn_cost")

            ## DEBUGGING ##
            # with tf.device("/cpu:0"):
            #     self.global_step = tf.Variable(0, name='global_step', trainable=False)
            #     self.rmsprop_min = RProp.RPropOptimizer().minimize(self.cost, global_step=self.global_step)
    
            # self.learning_rate_annealing_rate = (self.params['learning_rate_start'] - self.params['learning_rate_end']) / self.params['learning_rate_decay']
            # self.learning_rate = tf.maximum(self.params['learning_rate_end'],
            #                                 self.params['learning_rate_start'] - (self.learning_rate_annealing_rate * tf.to_float(self.global_step)))
            # self.rmsprop_min = tf.train.RMSPropOptimizer(self.learning_rate,self.params['rms_decay'],0.0,self.params['rms_eps']).minimize(self.cost, global_step=self.global_step)
            ##  END DEBUGGING ##
            
            self.rmsprop_min = tf.train.RMSPropOptimizer(self.params['lr'],self.params['rms_decay'],0.0,self.params['rms_eps']).minimize(self.cost)

        self.sess.run(tf.initialize_all_variables())
        tf.get_default_graph().finalize() # Disallow any more nodes to be added. Helps for debugging later
        print "###\n### Networks initialized\n### Ready to begin\n###"
    
    def set_all_weights(self, allweights):
        self.sess.run([
            self.assign_w1, self.assign_b1, self.assign_w2, self.assign_b2, self.assign_w3, self.assign_b3],
            feed_dict={ self.assign_w1_placeholder:allweights[0], self.assign_b1_placeholder:allweights[1],
                        self.assign_w2_placeholder:allweights[2], self.assign_b2_placeholder:allweights[3],
                        self.assign_w3_placeholder:allweights[4], self.assign_b3_placeholder:allweights[5]})

    def set_weights(self, weights, network_type=1):
        # print "############## Setting weights: layer {}, size of incoming weights: {} ({},{})".format(network_type,len(weights), weights[0].shape, weights[1].shape)
        if network_type == 1: # World
            if self.params['verbose'] >= 1:
                print "SETTING WEIGHTS FOR {}: {}".format(network_type, weights[0][0][0])
            self.sess.run([self.assign_w1, self.assign_b1], feed_dict={self.assign_w1_placeholder:weights[0], self.assign_b1_placeholder:weights[1]})
        elif network_type == 2: # Task
            self.sess.run([self.assign_w2, self.assign_b2], feed_dict={self.assign_w2_placeholder:weights[0], self.assign_b2_placeholder:weights[1]})
        elif network_type == 3: # Agent
            self.sess.run([self.assign_w3, self.assign_b3], feed_dict={self.assign_w3_placeholder:weights[0], self.assign_b3_placeholder:weights[1]})

    def stash_gradients(self, episode_delta):
        # print "Stashing Gradient: {}".format(episode_delta[0][0][0])
        self.num_grads_accumulated += 1
        for i, accumulate_grad in enumerate(self._gradient_list):
            accumulate_grad += (episode_delta[i])

    def clear_gradients(self):
        self.num_grads_accumulated = 0
        self._grad_w1 = np.zeros(([self.params['input_dims'], self.params['layer_1_hidden']])).astype(np.float32)
        self._grad_b1 = np.zeros(([self.params['layer_1_hidden']])).astype(np.float32)
        self._grad_w2 = np.zeros(([self.params['layer_1_hidden'], self.params['layer_2_hidden']])).astype(np.float32)
        self._grad_b2 = np.zeros(([self.params['layer_2_hidden']])).astype(np.float32)
        self._grad_w3 = np.zeros(([self.params['layer_2_hidden'], self.params['num_act']])).astype(np.float32)
        self._grad_b3 = np.zeros(([self.params['num_act']])).astype(np.float32)
        self._gradient_list = [ self._grad_w1, self._grad_b1, self._grad_w2, self._grad_b2, self._grad_w3, self._grad_b3]

    def get_and_clear_gradients(self):
        grads, num_grads_summed = self._gradient_list, self.num_grads_accumulated
        self.clear_gradients()
        start = np.sum(grads[0])
        ###      Average the gradients now ###
        for grad in grads:              # We only want the gradient average, not the giant number
            grad /= num_grads_summed    #(should also help with exploding gradients)
        ### End: Average the gradients now ###
        # if self.params['verbose'] >= 1:
        if True:
            print "Returning grads sum(grad[0]) of {} / {} = {}".format(start, num_grads_summed, np.sum(grads[0]))
        return grads

    def train(self, states, actions, rewards, terminals, next_states, display=False):
        q_target_max = np.amax(self.q(next_states), axis=1) # Pick the next state's best value to use in the reward (curRew + discount*(nextRew))

        start_weights = self.sess.run(self.all_layers)
        feed_dict={self.x: states, self.q_t: q_target_max, self.actions: actions, self.rewards: rewards, self.terminals:terminals}
        _, costs = self.sess.run([self.rmsprop_min, self.cost], feed_dict=feed_dict)
        end_weights = self.sess.run(self.all_layers)

        self.stash_gradients([end_weights[i] - start_weights[i] for i in range(len(start_weights))])
        if not self.params['allow_local_nn_weight_updates']:
            self.set_all_weights(start_weights)

        return costs
        
    def q(self, states):
        return self.sess.run(self.y, feed_dict={self.x: states})
        
    def save_weights(self, name, boolean_for_something):
        # SAVE WEIGHTS?!?  CLIENTS DON'T GET TO SAVE WEIGHTS.  THOSE COME FROM THE SERVER!
        pass
        