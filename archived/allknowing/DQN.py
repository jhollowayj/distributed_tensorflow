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

            self.rmsprop_min = tf.train.RMSPropOptimizer(self.params['lr'],self.params['rms_decay'],0.0,self.params['rms_eps']).minimize(self.cost)

        self.sess.run(tf.initialize_all_variables())
        tf.get_default_graph().finalize() # Disallow any more nodes to be added. Helps for debugging later
        print "###\n### Networks initialized\n### Ready to begin\n###"

    def train(self, bat_s, bat_a, bat_r, bat_t, bat_n, display=False):
        '''Please note: This allows the networks to change their weights!'''
        state = bat_s
        actions = bat_a
        rewards = bat_r
        terminals = bat_t
        next_state = bat_n

        q_target = self.sess.run(self.y,feed_dict={self.x: next_state})
        q_target_max = np.amax(q_target, axis=1)
        
        # starts = self.sess.run(self.all_layers)
        feed_dict={self.x: bat_s, self.q_t: q_target_max, self.actions: bat_a, self.rewards: bat_r, self.terminals:bat_t}
        _, costs = self.sess.run([self.rmsprop_min, self.cost], feed_dict=feed_dict)
        # ends = self.sess.run(self.all_layers)
        
        test = self.sess.run(self.y,feed_dict={self.x: state})
        # q_target = rewards + ((1.0 - terminals) * (self.params['discount'] * q_target_max))
        if display:
            for i,s in enumerate(bat_s):
                if s[0] == 1 and s[1] == 2:
                    print bat_s[i], bat_a[i], bat_r[i], bat_t[i], bat_n[i], "  \t", test[i], q_target_max[i], "\t", np.argmax(q_target[i])
                
        
        # self.stash_gradients([ends[i] - starts[i] for i in range(len(starts))])
        # if not self.params['allow_local_nn_weight_updates']:
        #     self.set_all_weights(starts)

        return costs
        
    def q(self, bat_s):
        return self.sess.run(self.y, feed_dict={self.x: bat_s})
        
    def save_weights(self, name, boolean_for_something):
        # w1, b2 = self.sess.run([self.w1, self.b2])
        return
