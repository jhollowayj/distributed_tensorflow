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

    def build_network_variables(self):
        # CREATE Storage Variables on the various Parameter Servers
        with tf.variable_scope("global"):
            self.w1s, self.b1s, self.w2s, self.b2s, self.w3s, self.b3s = {}, {}, {}, {}, {}, {}
            size_in, size_L1, size_L2, size_out = self.params['input_dims'], self.params['layer_1_hidden'],\
                                                  self.params['layer_2_hidden'], self.params['num_act']
            for i in range(1,4):
                with tf.device("/job:ps/task:0"):
                    with tf.variable_scope("world_{}".format(i)):
                        self.w1s[i] = tf.get_variable("weight", shape=(size_in, size_L1), dtype=tf.float32, initializer=tf.truncated_normal_initializer())
                        self.b1s[i] = tf.get_variable("bias", shape=(size_L1), dtype=tf.float32, initializer=tf.constant_initializer(0.1))
                
                with tf.device("/job:ps/task:1"):
                    with tf.variable_scope("task_{}".format(i)):
                        self.w2s[i] = tf.get_variable("weight", shape=(size_L1, size_L2), dtype=tf.float32, initializer=tf.truncated_normal_initializer())
                        self.b2s[i] = tf.get_variable("bias", shape=(size_L2), dtype=tf.float32, initializer=tf.constant_initializer(0.1))
                
                with tf.device("/job:ps/task:2"):
                    with tf.variable_scope("agent_{}".format(i)):
                        self.w3s[i] = tf.get_variable("weight", shape=(size_L2, size_out), dtype=tf.float32, initializer=tf.truncated_normal_initializer())
                        self.b3s[i] = tf.get_variable("bias", shape=(size_out), dtype=tf.float32, initializer=tf.constant_initializer(0.1))
            self.global_weights = [self.w1s, self.b1s, self.w2s, self.b2s, self.w3s, self.b3s]

    def build_worker_specific_model(self, worker_id):
        w, t, a, = self.params['wid'], self.params['tid'], self.params['aid']
        
        # Build local variables on each computer, sync with server in other functions.
        with tf.device("/job:worker/task:{}/cpu:0".format(worker_id)): # I might want to do "/worker/task{}/cpu:0".format(taskid) instead...
            with tf.device("/job:worker/task:{}/gpu:0".format(worker_id)): # TODO add in soft placement when you can...
                with tf.variable_scope("worker{}/local".format(worker_id)): # todo figure out what name to use here.
                # with tf.variable_scope("worker_{}_w{}.t{}.a{}".format(worker_id, w,t,a)):
                    self.__build_local_graph() # it was getting too deep...

    def __build_local_graph(self):
        with tf.name_scope("placeholders"):
            self.x = tf.placeholder(tf.float32,[None, self.params['input_dims']], name="nn_x")
            self.q_t = tf.placeholder(tf.float32,[None], name="nn_q_t")

            self.actions = tf.placeholder(tf.float32, [None, self.params['num_act']], name="nn_actions")
            self.rewards = tf.placeholder(tf.float32, [None], name="nn_rewards")
            self.terminals = tf.placeholder(tf.float32, [None], name="nn_terminals")

        discount = tf.constant(self.params['discount'], name="discount") # only need one constant.  :)

        # Just zeros, since we'll reset their weights from the PS variables anyway...
        self.local_weights = []
        with tf.variable_scope("world_{}".format(self.params['wid'])):
            w = tf.Variable(tf.zeros_like(self.w1s[self.params['wid']], dtype=tf.float32, name="weight"))
            b = tf.Variable(tf.zeros_like(self.b1s[self.params['wid']], dtype=tf.float32, name="bias"))
            o = tf.nn.relu(tf.add(tf.matmul(self.x,w), b), name="output")
            self.local_weights.append(w)
            self.local_weights.append(b)
        # TASK
        with tf.variable_scope("task_{}".format(self.params['tid'])):
            w = tf.Variable(tf.zeros_like(self.w2s[self.params['tid']], dtype=tf.float32, name="weight"))
            b = tf.Variable(tf.zeros_like(self.b2s[self.params['tid']], dtype=tf.float32, name="bias"))
            o = tf.nn.relu(tf.add(tf.matmul(o,w), b), name="output")
            self.local_weights.append(w)
            self.local_weights.append(b)
        # AGENT
        with tf.variable_scope("agent_{}".format(self.params['aid'])):
            w = tf.Variable(tf.zeros_like(self.w3s[self.params['aid']], dtype=tf.float32, name="weight"))
            b = tf.Variable(tf.zeros_like(self.b3s[self.params['aid']], dtype=tf.float32, name="bias"))
            self.y = tf.add(tf.matmul(o,w), b, name="output_y")
            self.local_weights.append(w)
            self.local_weights.append(b)

        #TODO Grab start weights
        #TODO Grab end weights
        #TODO Grab Delta of end-start
        #TODO Stash Delta somewhere?  OR do I just sync every time?  # SYNC every time
        start_vars = [tf.Variable(local_w, name="BLANK_STARTVAR") for local_w in self.local_weights]
        assign_start = [var.assign(local_w) for local_w, var in zip(self.local_weights, start_vars)] 
        
        #Q, Cost, Optimizer, etc.
        with tf.variable_scope("optimizer"):
            self.yj = tf.add(self.rewards, tf.mul(1.0-self.terminals, tf.mul(discount, self.q_t)), name="true_y")
            self.Q_pred = tf.reduce_sum(tf.mul(self.y, self.actions), reduction_indices=1, name="q_pred")
            self.cost = tf.reduce_sum(tf.pow(tf.sub(self.yj, self.Q_pred), 2), name="cost")
            with tf.control_dependencies(assign_start):
                self.rmsprop_min = tf.train.RMSPropOptimizer(
                    learning_rate=self.params['lr'],
                    decay=self.params['rms_decay'],
                    momentum=self.params['rms_momentum'],
                    epsilon=self.params['rms_eps']).minimize(self.cost) # control assign_start

        # Dont forget to average these to be lots of little steps across each client running it.
        with tf.control_dependencies([self.rmsprop_min]):
            self.local_deltas = [tf.sub(end_v, start_v, name="LOCAL_DELTA_LIST") for (end_v, start_v) in zip(start_vars, self.local_weights)] # control min
            ### build_sync_functions ###
            global_ids = [self.params['wid'], self.params['wid'], self.params['tid'], self.params['tid'], self.params['aid'], self.params['aid']]
            with tf.control_dependencies(self.local_deltas):
                self.sync_pushup_ops = [global_w[id].assign_add(local_d) for (local_d, global_w, id) in zip(self.local_deltas,  self.global_weights, global_ids)] 
            with tf.control_dependencies(self.sync_pushup_ops):
                self.sync_pulldown_ops = [local_w.assign(global_w[id])   for (local_w, global_w, id) in zip(self.local_weights, self.global_weights, global_ids)] 
        
        # WE could maybe do something with tf.get_Variable(global/world1/weight).assign(tf.get_variable(local/world1/weight))
        # maybe using the name, and replace local with global, etc.
        # It certainly needs to be more dynamic!  That's for sure.
        
    def set_session(self, session, global_step_inc, global_step_var):
        self.sess = session
        self.global_step_inc = global_step_inc 
        self.global_step_var = global_step_var 

    def train(self, states, actions, rewards, terminals, next_states, allow_update=True):
        q_target_max = np.amax(self.q(next_states), axis=1) # Pick the next state's best value to use in the reward (curRew + discount*(nextRew))

        feed_dict={self.x: self.scale_state_input(states), self.q_t: q_target_max, self.actions: actions, self.rewards: rewards, self.terminals:terminals}
        _, step, _, costs = self.sess.run([self.global_step_inc, self.global_step_var,
                                           self.rmsprop_min, self.cost, self.sync], feed_dict=feed_dict)
                                           

        return costs, step
        
    def q(self, states):
        return self.sess.run(self.y, feed_dict={self.x: self.scale_state_input(states)})
    
    def scale_state_input(self, state_to_scale): # TODO move this to be part of tensorflow to speed things up
        if self.params['input_scaling_vector'] is None:
            return state_to_scale
        else:
            return np.array(state_to_scale) / self.params['input_scaling_vector']
