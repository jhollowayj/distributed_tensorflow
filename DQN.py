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
        self.w1s, self.b1s, self.w2s, self.b2s, self.w3s, self.b3s = {}, {}, {}, {}, {}, {}
        for i in range(1,4):
            with tf.device("/job:ps/task:0"):
                with tf.variable_scope("world_{}".format(i)):
                    self.w1s[i] = tf.get_variable("weight", shape=(self.params['input_dims'], self.params['layer_1_hidden']), dtype=tf.float32, initializer=tf.truncated_normal_initializer())
                    self.b1s[i] = tf.get_variable("bias", shape=(self.params['layer_1_hidden']), dtype=tf.float32, initializer=tf.constant_initializer(0.1))
            
            with tf.device("/job:ps/task:1"):
                with tf.variable_scope("task_{}".format(i)):
                    self.w2s[i] = tf.get_variable("weight", shape=(self.params['layer_1_hidden'], self.params['layer_2_hidden']), dtype=tf.float32, initializer=tf.truncated_normal_initializer())
                    self.b2s[i] = tf.get_variable("bias", shape=(self.params['layer_2_hidden']), dtype=tf.float32, initializer=tf.constant_initializer(0.1))
            
            with tf.device("/job:ps/task:2"):
                with tf.variable_scope("agent_{}".format(i)):
                    self.w3s[i] = tf.get_variable("weight", shape=(self.params['layer_2_hidden'], self.params['num_act']), dtype=tf.float32, initializer=tf.truncated_normal_initializer())
                    self.b3s[i] = tf.get_variable("bias", shape=(self.params['num_act']), dtype=tf.float32, initializer=tf.constant_initializer(0.1))
        self.global_weights = [self.w1s, self.b1s, self.w2s, self.b2s, self.w3s, self.b3s]

    def build_worker_specific_model(self, worker_id):
        w, t, a, = self.params['wid'], self.params['tid'], self.params['aid']
        
        # Build local variables on each computer, sync with server in other functions.
        with tf.device("/cpu:0"): # I might want to do "/worker/task{}/cpu:0".format(taskid) instead...
            with tf.device("/gpu:0"): # TODO add in soft placement when you can...
                with tf.variable_scope("worker_{}_w{}.t{}.a{}".format(worker_id, w,t,a)):
                    self.__build_local_graph(): # it was getting too deep...
        self.__build_sync_functions()

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
        with tf.variable_scope("world_{}".format(w)):
            w = tf.Variable(tf.zeros(shape=tf.shape(self.w1s[self.params['wid']]), dtype=tf.float32, name="weight"))
            b = tf.Variable(tf.zeros(shape=tf.shape(self.b1s[self.params['wid']]), dtype=tf.float32, name="bias"))
            o = tf.nn.relu(tf.add(tf.matmul(self.x,w), b), name="output")
            weights.append(w)
            weights.append(b)
        # TASK
        with tf.variable_scope("task_{}".format(t)):
            w = tf.Variable(tf.zeros(shape=tf.shape(self.w2s[self.params['tid']]), dtype=tf.float32, name="weight"))
            b = tf.Variable(tf.zeros(shape=tf.shape(self.b2s[self.params['tid']]), dtype=tf.float32, name="bias"))
            o = tf.nn.relu(tf.add(tf.matmul(o,w), b), name="output")
            weights.append(w)
            weights.append(b)
        # AGENT
        with tf.variable_scope("agent_{}".format(a)):
            w = tf.Variable(tf.zeros(shape=tf.shape(self.w3s[self.params['aid']]), dtype=tf.float32, name="weight"))
            b = tf.Variable(tf.zeros(shape=tf.shape(self.b3s[self.params['aid']]), dtype=tf.float32, name="bias"))
            self.y = tf.add(tf.matmul(o,w), b, name="output_y")
            weights.append(w)
            weights.append(b)

        #TODO Grab start weights
        #TODO Grab end weights
        #TODO Grab Delta of end-start
        #TODO Stash Delta somewhere?  OR do I just sync every time?  # SYNC every time
        start_vars = [tf.identity(local_w) for local_w in self.local_weights]
        end_vars   = [tf.identity(local_w) for local_w in self.local_weights]
        
        #Q, Cost, Optimizer, etc.
        with tf.variable_scope("optimizer"):
            self.yj = tf.add(self.rewards, tf.mul(1.0-self.terminals, tf.mul(discount, self.q_t)), name="true_y")
            self.Q_pred = tf.reduce_sum(tf.mul(self.y, self.actions), reduction_indices=1, name="q_pred")
            self.cost = tf.reduce_sum(tf.pow(tf.sub(self.yj, self.Q_pred), 2), name="cost")
            self.rmsprop_min = tf.train.RMSPropOptimizer(
                learning_rate=self.params['lr'],
                decay=self.params['rms_decay'],
                momentum=self.params['rms_momentum'],
                epsilon=self.params['rms_eps']).minimize(self.cost)
    
        ### build_sync_functions ###
        w, t, a = self.params['wid'], self.params['tid'], self.params['aid']
        global_ids = [w, w, t, t, a, a]
        self.sync_pushup_ops = [global_w[id].assign_add(local_d) for (local_d, global_w, id) in zip(self.local_deltas,  self.global_weights, global_ids)] 
        self.sync_pulldown_ops = [local_w.assign(global_w[id])   for (local_w, global_w, id) in zip(self.local_weights, self.global_weights, global_ids)] 
        # WE could maybe do something with tf.get_Variable(global/world1/weight).assign(tf.get_variable(local/world1/weight))
        # maybe using the name, and replace local with global, etc.
        # It certainly needs to be more dynamic!  That's for sure.
        
    def sync_with_param_server(self):
        pass
        
    def set_session(self, session, global_step_inc, global_step_var):
        self.sess = session
        self.global_step_inc = global_step_inc 
        self.global_step_var = global_step_var 

    def train(self, states, actions, rewards, terminals, next_states, allow_update=True):
        q_target_max = np.amax(self.q(next_states), axis=1) # Pick the next state's best value to use in the reward (curRew + discount*(nextRew))

        feed_dict={self.x: self.scale_state_input(states), self.q_t: q_target_max, self.actions: actions, self.rewards: rewards, self.terminals:terminals}
        _, step, _, costs = self.sess.run([self.global_step_inc, self.global_step_var,
                                           self.rmsprop_min, self.cost], feed_dict=feed_dict)

        return costs, step
        
    def q(self, states):
        return self.sess.run(self.y, feed_dict={self.x: self.scale_state_input(states)})
    
    def scale_state_input(self, state_to_scale): # TODO move this to be part of tensorflow to speed things up
        if self.params['input_scaling_vector'] is None:
            return state_to_scale
        else:
            return np.array(state_to_scale) / self.params['input_scaling_vector']
