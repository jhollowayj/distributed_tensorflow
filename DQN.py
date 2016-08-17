import numpy as np
import tensorflow as tf
import time

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

    def build_global_variables(self):
        # CREATE Storage Variables on the various Parameter Servers
        with tf.variable_scope("global"):
            self.w1s, self.b1s, self.w2s, self.b2s, self.w3s, self.b3s = {}, {}, {}, {}, {}, {}
            self.ph_w1s, self.ph_b1s, self.ph_w2s, self.ph_b2s, self.ph_w3s, self.ph_b3s  = {}, {}, {}, {}, {}, {}
            
            w1_shape, b1_shape = [self.params['input_dims'], self.params['layer_1_hidden']],  [self.params['layer_1_hidden']]
            w2_shape, b2_shape = [self.params['layer_1_hidden'], self.params['layer_2_hidden']],  [self.params['layer_2_hidden']]
            w3_shape, b3_shape = [self.params['layer_2_hidden'], self.params['num_act']], [self.params['num_act']]
            
            for i in range(1,4): # 1,2,3
                with tf.device("/job:ps/task:0"):
                    with tf.variable_scope("world_{}".format(i)):
                        self.w1s[i] = tf.get_variable("weight", shape=w1_shape, dtype=tf.float32, initializer=tf.truncated_normal_initializer())
                        self.b1s[i] = tf.get_variable("bias",   shape=b1_shape, dtype=tf.float32, initializer=tf.constant_initializer(0.1))
                        self.ph_w1s[i] = tf.placeholder(tf.float32, shape=w1_shape)
                        self.ph_b1s[i] = tf.placeholder(tf.float32, shape=b1_shape)
                
                with tf.device("/job:ps/task:1"):
                    with tf.variable_scope("task_{}".format(i)):
                        self.w2s[i] = tf.get_variable("weight", shape=w2_shape, dtype=tf.float32, initializer=tf.truncated_normal_initializer())
                        self.b2s[i] = tf.get_variable("bias",   shape=b2_shape, dtype=tf.float32, initializer=tf.constant_initializer(0.1))
                        self.ph_w2s[i] = tf.placeholder(tf.float32, shape=w2_shape)
                        self.ph_b2s[i] = tf.placeholder(tf.float32, shape=b2_shape)
                
                with tf.device("/job:ps/task:2"):
                    with tf.variable_scope("agent_{}".format(i)):
                        self.w3s[i] = tf.get_variable("weight", shape=w3_shape, dtype=tf.float32, initializer=tf.truncated_normal_initializer())
                        self.b3s[i] = tf.get_variable("bias",   shape=b3_shape, dtype=tf.float32, initializer=tf.constant_initializer(0.1))
                        self.ph_w3s[i] = tf.placeholder(tf.float32, shape=w3_shape)
                        self.ph_b3s[i] = tf.placeholder(tf.float32, shape=b3_shape)
                
            self.global_weights = [self.w1s, self.b1s, self.w2s, self.b2s, self.w3s, self.b3s]
        return [x[y] for x in self.global_weights for y in x] # Flatten dictionaries into one list

    def build_worker_specific_model(self, worker_id, local_graph):
        self.sess_local = tf.Session(graph=local_graph)
        
        # Build local variables on each computer, sync with server in other functions.
        with tf.variable_scope("worker{}/local".format(worker_id)):
            with tf.device("/gpu:0"):
                with tf.name_scope("placeholders"):
                    self.x = tf.placeholder(tf.float32,[None, self.params['input_dims']], name="nn_x")
                    self.q_t = tf.placeholder(tf.float32,[None], name="nn_q_t")

                    self.actions = tf.placeholder(tf.float32, [None, self.params['num_act']], name="nn_actions")
                    self.rewards = tf.placeholder(tf.float32, [None], name="nn_rewards")
                    self.terminals = tf.placeholder(tf.float32, [None], name="nn_terminals")

                discount = tf.constant(self.params['discount'], name="discount") # only need one constant.  :)

                w1_shape, b1_shape = [self.params['input_dims'], self.params['layer_1_hidden']],  [self.params['layer_1_hidden']]
                w2_shape, b2_shape = [self.params['layer_1_hidden'], self.params['layer_2_hidden']],  [self.params['layer_2_hidden']]
                w3_shape, b3_shape = [self.params['layer_2_hidden'], self.params['num_act']], [self.params['num_act']]
        
                # Just zeros, since we'll reset their weights from the PS variables anyway...
                self.local_weights = []
                with tf.variable_scope("world_{}".format(self.params['wid'])):
                    w = tf.Variable(tf.zeros(shape=w1_shape, dtype=tf.float32), name="weight")
                    b = tf.Variable(tf.zeros(shape=b1_shape, dtype=tf.float32), name="bias")
                    o = tf.nn.relu(tf.add(tf.matmul(self.x,w), b), name="output")
                    self.local_weights.append(w)
                    self.local_weights.append(b)
                # # TASK
                with tf.variable_scope("task_{}".format(self.params['tid'])):
                    w = tf.Variable(tf.zeros(shape=w2_shape, dtype=tf.float32), name="weight")
                    b = tf.Variable(tf.zeros(shape=b2_shape, dtype=tf.float32), name="bias")
                    o = tf.nn.relu(tf.add(tf.matmul(o,w), b), name="output")
                    self.local_weights.append(w)
                    self.local_weights.append(b)
                # # AGENT
                with tf.variable_scope("agent_{}".format(self.params['aid'])):
                    w = tf.Variable(tf.zeros(shape=w3_shape, dtype=tf.float32), name="weight")
                    b = tf.Variable(tf.zeros(shape=b3_shape, dtype=tf.float32), name="bias")
                    self.y = tf.add(tf.matmul(o,w), b, name="output_y")
                    self.local_weights.append(w)
                    self.local_weights.append(b)

                # self.sess_local.run([x.initializer for x in self.local_weights]) #  + self.start_vars
                print w.get_shape()
                self.start_vars = [tf.Variable(tf.zeros(shape=local_w.get_shape()), name="STARTVAR")  for local_w in self.local_weights]
                self.delta_vars = [tf.Variable(tf.zeros(shape=local_w.get_shape()), name="VAR_DELTA") for local_w in self.local_weights]
                assign_start = [var.assign(local_w) for local_w, var in zip(self.local_weights, self.start_vars)] 
                
                # Q, Cost, Optimizer, etc.
                with tf.variable_scope("optimizer"):
                    self.yj = tf.add(self.rewards, tf.mul(1.0-self.terminals, tf.mul(discount, self.q_t)), name="true_y")
                    self.Q_pred = tf.reduce_sum(tf.mul(self.y, self.actions), reduction_indices=1, name="q_pred")
                    self.cost = tf.reduce_sum(tf.pow(tf.sub(self.yj, self.Q_pred), 2), name="cost")
                    with tf.control_dependencies(assign_start):
                        self.rmsprop_min = tf.train.RMSPropOptimizer(
                            learning_rate=self.params['lr'],
                            decay=self.params['rms_decay'],
                            momentum=self.params['rms_momentum'],
                            epsilon=self.params['rms_eps']).minimize(self.cost)

                        with tf.control_dependencies([self.rmsprop_min]):
                            assign_deltas = [delta_v.assign(tf.sub(end_v, start_v))\
                                                for (delta_v, end_v, start_v)
                                                in zip(self.delta_vars, self.start_vars, self.local_weights)]
                            
                self.sess_local.run(tf.initialize_all_variables())
                
    def send_gradients(self):
        global_ids = [self.params['wid'], self.params['wid'], self.params['tid'], self.params['tid'], self.params['aid'], self.params['aid']]
        for (local_d, global_w, id) in zip(self.local_deltas,  self.global_weights, global_ids):
            self.sess_global.run(local_)
        self.sync_pushup_ops = [global_w[id].assign_add(local_d) ] 
        
    def update_weights(self):
        global_ids = [self.params['wid'], self.params['wid'], self.params['tid'], self.params['tid'], self.params['aid'], self.params['aid']]
        with tf.control_dependencies(self.sync_pushup_ops):
            self.sync_pulldown_ops = [local_w.assign(global_w[id])   for (local_w, global_w, id) in zip(self.local_weights, self.global_weights, global_ids)] 
        
    def set_global_session(self, global_session, global_step_inc, global_step_var):
        self.sess_global = global_session
        self.global_step_inc = global_step_inc 
        self.global_step_var = global_step_var 

    def train(self, states, actions, rewards, terminals, next_states, allow_update=True):
        q_target_max = np.amax(self.q(next_states), axis=1) # Pick the next state's best value to use in the reward (curRew + discount*(nextRew))

        feed_dict={self.x: self.scale_state_input(states), self.q_t: q_target_max, self.actions: actions, self.rewards: rewards, self.terminals:terminals}
        
        result_local = self.sess_local.run([self.cost, self.rmsprop_min] + assign_deltas, feed_dict=feed_dict)
        _, gstep = self.sess_global.run([self.global_step_inc, self.global_step_var])
        
        # TODO sync weights
        costs = result_local[0]
        return costs, step
        
    def q(self, states):
        return self.sess_local.run(self.y, feed_dict={self.x: self.scale_state_input(states)})
    
    def scale_state_input(self, state_to_scale): # TODO move this to be part of tensorflow to speed things up
        if self.params['input_scaling_vector'] is None:
            return state_to_scale
        else:
            return np.array(state_to_scale) / self.params['input_scaling_vector']
