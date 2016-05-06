import numpy as np
import tensorflow as tf
import time
class DQN:
    def __init__(self, input_dims = 2, num_act = 4,
            eps = 1.0, discount = 0.95, lr =  0.0002,
            rms_eps = 1e-6, rms_decay=0.99,
            allow_local_nn_weight_updates=False,
            requested_gpu_vram_percent=0.01,
            device_to_use=0):
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
        }
        if self.params['allow_local_nn_weight_updates']:
            print "NOTE: network is allowed to update his own weights!" 

        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=self.params['requested_gpu_vram_percent'])
        self.sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
        
        with tf.device({-1: "/cpu:0", 0: "/gpu:0", 1: "/gpu:1"}[device_to_use]):
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

            ### Gradients ###
            self.clear_gradients()
            self.compgrad_vars = [ self.w1, self.b2, self.w2, self.b2, self.w3, self.b3 ]
            ### end Gradients ###
            
            #Q,Cost,Optimizer
            self.discount = tf.constant(self.params['discount'], name="nn_discount")
            self.yj = tf.add(self.rewards, tf.mul(1.0-self.terminals, tf.mul(self.discount, self.q_t)), name="nn_yj_whatever-that-is")
            self.Q_pred = tf.reduce_sum(tf.mul(self.y,self.actions), reduction_indices=1, name="nn_q_pred")
            self.cost = tf.reduce_sum(tf.pow(tf.sub(self.yj, self.Q_pred), 2), name="nn_cost")

            self.rmsprop = tf.train.RMSPropOptimizer(self.params['lr'],self.params['rms_decay'],0.0,self.params['rms_eps']).minimize(self.cost) # Orig
            # # Failed attempt.   :(  8 hours later... we go back to the original idea...
            # self.rmsprop = tf.train.RMSPropOptimizer(self.params['lr'],self.params['rms_decay'],0.0,self.params['rms_eps'])
            # self.comp_grad = self.rmsprop.compute_gradients(self.cost)
            # self.app_grads = self.rmsprop.apply_gradients(self.comp_grad)

            self.rmsprop = tf.train.RMSPropOptimizer(self.params['lr'],self.params['rms_decay'],0.0,self.params['rms_eps'])
            self.comp_grads = self.rmsprop.compute_gradients(self.cost)
            self.grad_placeholder = [(tf.placeholder("float", shape=grad[1].get_shape(), name="grad_placeholder"), grad[1]) for grad in self.comp_grads]
            self.apply_grads = self.rmsprop.apply_gradients(self.grad_placeholder)
            
            # # Used >> # results = self.sess.run([grad for grad, _ in self.comp_grad] + [self.app_grads, self.cost], feed_dict=feed_dict)
            # # grads_and_vars, _, costs = self.sess.run([self.comp_grad, self.app_grads, self.cost], feed_dict=feed_dict) # Try two!
            
        init = tf.initialize_all_variables()
        self.sess.run(init)
        print "###\n### Initialized MrKulk's network\n###"
    
    def set_weights(self, weights, network_type=1):
        # print "############## Setting weights: layer {}, size of incoming weights: {} ({},{})".format(network_type,len(weights), weights[0].shape, weights[1].shape)
        todo = []
        if network_type == 1: # World
            print "SETTING WEIGHTS FOR {}: {}".format(network_type, weights[0][0][0])
            todo.append(self.w1.assign(weights[0]))
            todo.append(self.b1.assign(weights[1]))
        elif network_type == 2: # Task
            todo.append(self.w2.assign(weights[0]))
            todo.append(self.b2.assign(weights[1]))
        elif network_type == 3: # Agent
            todo.append(self.w3.assign(weights[0]))
            todo.append(self.b3.assign(weights[1]))
        self.sess.run(todo)
    def stash_gradients(self, episode_delta):
        # print "Stashing Gradient: {}".format(episode_delta[0][0][0])
        self.num_grads_accumulated += 1
        for i, accumulate_grad in enumerate(self._gradient_list):
            accumulate_grad += (episode_delta[i] * self.params['lr'])

    def clear_gradients(self):
        self.num_grads_accumulated = 0
        self._grad_w1 = np.zeros(([self.params['input_dims'], self.params['layer_1_hidden']])).astype(np.float32)
        self._grad_b1 = np.zeros(([self.params['layer_1_hidden']])).astype(np.float32)
        self._grad_w2 = np.zeros(([self.params['layer_1_hidden'], self.params['layer_2_hidden']])).astype(np.float32)
        self._grad_b2 = np.zeros(([self.params['layer_2_hidden']])).astype(np.float32)
        self._grad_w3 = np.zeros(([self.params['layer_2_hidden'], self.params['num_act']])).astype(np.float32)
        self._grad_b3 = np.zeros(([self.params['num_act']])).astype(np.float32)
        self._gradient_list = [ self._grad_w1, self._grad_b1, self._grad_w2, self._grad_b2, self._grad_w3, self._grad_b3]

    def _get_gradients(self): # removed to help with the average gradients code added below
        # print self._gradient_list[0][0][0] # Debugging
        return self._gradient_list, self.num_grads_accumulated
        
    def get_and_clear_gradients(self):
        grads, num_grads_summed = self._get_gradients()
        self.clear_gradients()
        start = grads[0][0][0]
        ###      Average the gradients now ###
        for grad in grads:              # We only want the gradient average, not the giant number
            grad /= num_grads_summed    #(should also help with exploding gradients)
        ### End: Average the gradients now ###
        print "Returning grads[0][0][0] of {} / {} = {}".format(start, num_grads_summed, grads[0][0][0])
        return grads

    def train(self,bat_s,bat_a,bat_t,bat_n,bat_r):
        feed_dict={self.x: bat_s, self.q_t: np.zeros(bat_n.shape[0]), self.actions: bat_a, self.terminals:bat_t, self.rewards: bat_r}
        q_t = self.sess.run(self.y,feed_dict=feed_dict)
        q_t = np.amax(q_t,axis=1)
        feed_dict={self.x: bat_s, self.q_t: q_t, self.actions: bat_a, self.terminals:bat_t, self.rewards: bat_r}
        # start = self.sess.run(self.w1)[0][0]

        # # t = time.time()
        # start_weights = self.sess.run(self.compgrad_vars) # Grab initial weights
        # _, costs = self.sess.run([self.rmsprop, self.cost], feed_dict=feed_dict) # change weights
        # # grads, _, costs = self.sess.run([self.comp_grads, self.apply_grads, self.cost])
        # end_weights = self.sess.run(self.compgrad_vars)   # Grab final weights
        # # t2 = time.time()
        # # print "Took {} seconds...".format(t2-t)
           
        grad_vals = self.sess.run([grad[0] for grad in self.comp_grads], feed_dict=feed_dict)
        self.stash_gradients(grad_vals)
            
        
        if self.params['allow_local_nn_weight_updates']:
            feed_dict2={self.x: bat_s, self.q_t: q_t, self.actions: bat_a, self.terminals:bat_t, self.rewards: bat_r,
                    self.grad_placeholder[0][0]: grad_vals[0], # Change these lines to add in grad_placeholders to use the code
                    self.grad_placeholder[1][0]: grad_vals[1], # Found in this SO post here... http://stackoverflow.com/questions/34687761/ 
                    self.grad_placeholder[2][0]: grad_vals[2],
                    self.grad_placeholder[3][0]: grad_vals[3],
                    self.grad_placeholder[4][0]: grad_vals[4],
                    self.grad_placeholder[5][0]: grad_vals[5],
                    }
            self.sess.run(self.apply_grads, feed_dict=feed_dict2) # TODO TODO TODO currently returning None.  Not sure why, but that needs to be fixed.
        costs = self.sess.run(self.cost, feed_dict=feed_dict)

        # Grab and store the deltas to ship to the server
        # deltas = []
        # for i in range(len(start_weights)):
        #     deltas.append(end_weights[i] - start_weights[i]) # Calculate the difference in weights
        # self.stash_gradients(deltas) # Gradients, skipping app_grads result and cost result
        
        # if self.params['lock_network_changes']:
        #     # t = time.time()
        #     self.sess.run([
        #         self.w1.assign(start_weights[0], use_locking=True),
        #         self.b1.assign(start_weights[1], use_locking=True),
        #         self.w2.assign(start_weights[2], use_locking=True),
        #         self.b2.assign(start_weights[3], use_locking=True),
        #         self.w3.assign(start_weights[4], use_locking=True),
        #         self.b3.assign(start_weights[5], use_locking=True)
        #     ])
            # t2 = time.time()
            # print "Took {} seconds==============".format(t2-t)

        return costs

    def q(self, bat_s):
        return self.sess.run(self.y, feed_dict = {
            self.x: bat_s,
            self.q_t: np.zeros(1),
            self.actions: np.zeros((1, self.params['num_act'])),
            self.terminals:np.zeros(1),
            self.rewards: np.zeros(1)
        })
        
    def save_weights(self, name, boolean_for_something):
        # w1, b2 = self.sess.run([self.w1, self.b2])
        return
