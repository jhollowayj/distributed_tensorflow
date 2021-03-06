import numpy as np
import tensorflow as tf
import time
import sys

# CUDA_VISIBLE_DEVICES="0,1"
# Trying to solve the issue that prevents cuda from finding the gpus

class DQN:
    def __init__(self, wid=1, tid=1, aid=1,
            input_dims = 2, num_act = 4,
            eps = 1.0, discount = 0.90, lr = 0.0002,
            rms_eps = 1e-6, rms_decay=0.99, rms_momentum=0.0,
            input_scaling_vector=None, verbose = 0):
        self.params = {
            'wid' : wid,
            'tid' : tid,
            'aid' : aid,
            'input_dims': input_dims,
            'layer_1_hidden': 1000,
            'layer_2_hidden': 1000,
            'num_act': num_act,
            'lr': lr,
            'discount': discount,
            'rms_eps': rms_eps,
            'rms_decay': rms_decay,
            'rms_momentum': rms_momentum,
            'input_scaling_vector': None if input_scaling_vector is None else np.array(input_scaling_vector),
            'verbose': verbose,
            'learning_rate_start':  0.003,
            'learning_rate_end':    0.003, #0.000001,
            'learning_rate_decay':  3000,
        }
        self.prioritaztion_training = True

    def build_global_variables(self):
        # CREATE Storage Variables on the various Parameter Servers
        with tf.variable_scope("global"):
            self.w1s, self.b1s, self.w2s, self.b2s, self.w3s, self.b3s = {}, {}, {}, {}, {}, {}
            self.ph_w1s, self.ph_b1s, self.ph_w2s, self.ph_b2s, self.ph_w3s, self.ph_b3s  = {}, {}, {}, {}, {}, {}
            self.azz_w1s, self.azz_b1s, self.azz_w2s, self.azz_b2s, self.azz_w3s, self.azz_b3s  = {}, {}, {}, {}, {}, {}
            
            w1_shape, b1_shape = [self.params['input_dims'], self.params['layer_1_hidden']],  [self.params['layer_1_hidden']]
            w2_shape, b2_shape = [self.params['layer_1_hidden'], self.params['layer_2_hidden']],  [self.params['layer_2_hidden']]
            w3_shape, b3_shape = [self.params['layer_2_hidden'], self.params['num_act']], [self.params['num_act']]
            
            for i in range(1,4): # 1,2,3
                with tf.variable_scope("world_{}".format(i)):
                    self.w1s[i] = tf.get_variable("weight", shape=w1_shape, dtype=tf.float32, initializer=tf.contrib.layers.xavier_initializer())
                    self.b1s[i] = tf.get_variable("bias",   shape=b1_shape, dtype=tf.float32, initializer=tf.constant_initializer(0.1))
                    self.ph_w1s[i] = tf.placeholder(tf.float32, shape=w1_shape)
                    self.ph_b1s[i] = tf.placeholder(tf.float32, shape=b1_shape)
                    self.azz_w1s[i]= self.w1s[i].assign_add(self.ph_w1s[i])
                    self.azz_b1s[i]= self.b1s[i].assign_add(self.ph_b1s[i])
            
                with tf.variable_scope("task_{}".format(i)):
                    self.w2s[i] = tf.get_variable("weight", shape=w2_shape, dtype=tf.float32, initializer=tf.contrib.layers.xavier_initializer())
                    self.b2s[i] = tf.get_variable("bias",   shape=b2_shape, dtype=tf.float32, initializer=tf.constant_initializer(0.1))
                    self.ph_w2s[i] = tf.placeholder(tf.float32, shape=w2_shape)
                    self.ph_b2s[i] = tf.placeholder(tf.float32, shape=b2_shape)
                    self.azz_w2s[i]= self.w2s[i].assign_add(self.ph_w2s[i])
                    self.azz_b2s[i]= self.b2s[i].assign_add(self.ph_b2s[i])
            
                with tf.variable_scope("agent_{}".format(i)):
                    self.w3s[i] = tf.get_variable("weight", shape=w3_shape, dtype=tf.float32, initializer=tf.contrib.layers.xavier_initializer())
                    self.b3s[i] = tf.get_variable("bias",   shape=b3_shape, dtype=tf.float32, initializer=tf.constant_initializer(0.1))
                    self.ph_w3s[i] = tf.placeholder(tf.float32, shape=w3_shape)
                    self.ph_b3s[i] = tf.placeholder(tf.float32, shape=b3_shape)
                    self.azz_w3s[i]= self.w3s[i].assign_add(self.ph_w3s[i])
                    self.azz_b3s[i]= self.b3s[i].assign_add(self.ph_b3s[i])
                
            self.global_weights = [self.w1s, self.b1s, self.w2s, self.b2s, self.w3s, self.b3s]

        # Create lists of the specific objects this worker will want for syncing
        w, t, a = self.params['wid'], self.params['tid'], self.params['aid']
        self.global_vars          = [self.w1s[w],     self.b1s[w],     self.w2s[t],     self.b2s[t],     self.w3s[a],     self.b3s[a]]
        self.global_placeholders  = [self.ph_w1s[w],  self.ph_b1s[w],  self.ph_w2s[t],  self.ph_b2s[t],  self.ph_w3s[a],  self.ph_b3s[a]]
        self.global_assign_adders = [self.azz_w1s[w], self.azz_b1s[w], self.azz_w2s[t], self.azz_b2s[t], self.azz_w3s[a], self.azz_b3s[a]]
        
    def _dense(self, input, s1, s2, relu=True, assign_add=False):
        ''' returns output, list of weights, list of placeholders, and list of assign operations'''
            
        print input.get_shape(), s1, s2,
        
        w = tf.Variable(tf.zeros(shape=(s1, s2), dtype=tf.float32), name="weight")
        b = tf.Variable(tf.zeros(shape=(s2),     dtype=tf.float32), name="bias")
        o = tf.add(tf.matmul(input,w), b, name="output")
        if relu:
            o = tf.nn.relu(o, name="output_active")
        print o.get_shape()
        
        ph_w = tf.placeholder(tf.float32, shape=w.get_shape(), name="placeholder_weight")
        ph_b = tf.placeholder(tf.float32, shape=b.get_shape(), name="placeholder_bais")
        assigns = [w.assign(ph_w), b.assign(ph_b)] # [w.assign_add(ph_w), b.assign_add(ph_b)] if assign_add else **current**
        return o, [w, b], [ph_w, ph_b], assigns

    def build_worker_specific_model(self, worker_id, local_graph):
        self.sess_local = tf.Session(graph=local_graph)
        
        # Build local variables on each computer, sync with server in other functions.
        with tf.variable_scope("worker{}/local".format(worker_id)):
            with tf.device("/gpu:0"):
            
                size_in = self.params['input_dims']
                size_L1 = self.params['layer_1_hidden']
                size_L2 = self.params['layer_2_hidden']
                size_out= self.params['num_act']

                with tf.name_scope("placeholders"):
                    self.x = tf.placeholder(tf.float32,[None, size_in], name="nn_x")
                    self.q_t = tf.placeholder(tf.float32,[None], name="nn_q_t")

                    self.actions = tf.placeholder(tf.float32, [None, size_out], name="nn_actions")
                    self.rewards = tf.placeholder(tf.float32, [None], name="nn_rewards")
                    self.terminals = tf.placeholder(tf.float32, [None], name="nn_terminals")
        
                # Just zeros, since we'll reset their weights from the PS variables anyway...
                self.local_weights, self.local_weights_ph, self.local_weights_azz = [], [], []
                with tf.variable_scope("world_{}".format(self.params['wid'])):
                    o, w, ph, azz = self._dense(self.x, size_in, size_L1)
                    self.local_weights += w
                    self.local_weights_ph += ph
                    self.local_weights_azz += azz
                # # TASK
                with tf.variable_scope("task_{}".format(self.params['tid'])):
                    o, w, ph, azz = self._dense(o, size_L1, size_L2)
                    self.local_weights += w
                    self.local_weights_ph += ph
                    self.local_weights_azz += azz
                # # AGENT
                with tf.variable_scope("agent_{}".format(self.params['aid'])):
                    self.y, w, ph, azz = self._dense(o, size_L2, size_out)
                    self.local_weights += w
                    self.local_weights_ph += ph
                    self.local_weights_azz += azz

                self.start_vars = [tf.Variable(tf.zeros(shape=local_w.get_shape()), name="STARTVAR",  dtype=tf.float32) for local_w in self.local_weights]
                self.delta_vars = [tf.Variable(tf.zeros(shape=local_w.get_shape()), name="VAR_DELTA", dtype=tf.float32) for local_w in self.local_weights]
                assign_start = [var.assign(local_w) for local_w, var in zip(self.local_weights, self.start_vars)] 
                
                # Q, Cost, Optimizer, etc.
                with tf.variable_scope("optimizer"):
                    with tf.control_dependencies(assign_start):
                        discount = tf.constant(self.params['discount'], name="discount") # only need one constant.  :)
                        self.yj = tf.add(self.rewards, tf.mul(1.0-self.terminals, tf.mul(discount, self.q_t)), name="true_y")
                        self.Q_pred = tf.reduce_sum(tf.mul(self.y, self.actions), reduction_indices=1, name="q_pred")
                        self.cost = tf.reduce_mean(tf.pow(tf.sub(self.yj, self.Q_pred), 2), name="cost")
                        with tf.device("/gpu:0"):
                            self.rmsprop_min = tf.train.RMSPropOptimizer(
                                learning_rate=self.params['lr'],
                                decay=self.params['rms_decay'],
                                momentum=self.params['rms_momentum'],
                                epsilon=self.params['rms_eps']).minimize(self.cost)

                            with tf.control_dependencies([self.rmsprop_min]):
                                self.assign_deltas = [delta_v.assign(tf.sub(end_v, start_v))\
                                                    for   (delta_v,         start_v,         end_v)
                                                    in zip(self.delta_vars, self.start_vars, self.local_weights)]
                            
                self.sess_local.run(tf.initialize_all_variables())
                
    def send_gradients(self):
        feed_dict = {}
        for (local_d, global_ph) in zip(self.sess_local.run(self.delta_vars),  self.global_placeholders):
            feed_dict[global_ph] = local_d
        self.sess_global.run(self.global_assign_adders, feed_dict=feed_dict) # Assign add on global
        
    def update_weights(self):
        feed_dict = {}
        for (global_var, local_ph) in zip(self.sess_global.run(self.global_vars), self.local_weights_ph):
            feed_dict[local_ph] = global_var
        self.sess_local.run(self.local_weights_azz, feed_dict=feed_dict)
        
    def set_global_session(self, global_session, global_step_inc, global_step_var):
        self.sess_global = global_session
        self.global_step_inc = global_step_inc 
        self.global_step_var = global_step_var

    def train(self, states, actions, rewards, terminals, next_states, allow_update=True, loop_cnt=0):
        q_target_max = np.amax(self.q(next_states), axis=1) # Pick the next state's best value to use in the reward (curRew + discount*(nextRew))
        
        feed_dict={self.x: self.scale_state_input(states), self.q_t: q_target_max, self.actions: actions, self.rewards: rewards, self.terminals:terminals}
        result_local = self.sess_local.run([self.cost] + self.assign_deltas + [self.rmsprop_min], feed_dict=feed_dict)
        _, gstep = self.sess_global.run([self.global_step_inc, self.global_step_var]) # update train count
        
        if allow_update: # if eval, don't send, etc...
            self.send_gradients()
        self.update_weights()
        
        costs = result_local[0]
        if allow_update and self.prioritaztion_training:
            # Poor man's 'prioritization replay', but should work since we throw away the exp.db and cant replay it...
            if costs > 1.0 and loop_cnt < 50:
                print "Re-training erroneous dataset: c:{:<15f} at loop:{}".format(costs, loop_cnt+1) 
                sys.stdout.flush()
                return self.train(states, actions, rewards, terminals, next_states, allow_update, loop_cnt+1)
            else:
                return costs, gstep
        else:
            return costs, gstep

        
    def q(self, states):
        return self.sess_local.run(self.y, feed_dict={self.x: self.scale_state_input(states)})
    
    def scale_state_input(self, state_to_scale): # TODO move this to be part of tensorflow to speed things up
        if self.params['input_scaling_vector'] is None:
            return state_to_scale
        else:
            return np.array(state_to_scale) / self.params['input_scaling_vector']
