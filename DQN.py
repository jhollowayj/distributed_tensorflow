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

        self.build_network_variables()
        
    def build_network_variables(self):
        with tf.name_scope("placeholders"):
            self.x = tf.placeholder(tf.float32,[None, self.params['input_dims']], name="nn_x")
            self.q_t = tf.placeholder(tf.float32,[None], name="nn_q_t")

            self.actions = tf.placeholder(tf.float32, [None, self.params['num_act']], name="nn_actions")
            self.rewards = tf.placeholder(tf.float32, [None], name="nn_rewards")
            self.terminals = tf.placeholder(tf.float32, [None], name="nn_terminals")

        ### Network ###
        layer_1_hidden = self.params['layer_1_hidden']
        layer_2_hidden = self.params['layer_2_hidden']

        with tf.variable_scope("world_{}".format(self.params['wid'])):
            self.w1 = tf.get_variable("weight", shape=(self.params['input_dims'], layer_1_hidden), dtype=tf.float32, initializer=tf.truncated_normal_initializer())
            self.b1 = tf.get_variable("bias", shape=(layer_1_hidden), dtype=tf.float32, initializer=tf.constant_initializer(0.1))
            self.o1 = tf.nn.relu(tf.add(tf.matmul(self.x,self.w1),self.b1), name="output")

        with tf.variable_scope("task_{}".format(self.params['tid'])):
            self.w2 = tf.get_variable("weight", shape=(layer_1_hidden, layer_2_hidden), dtype=tf.float32, initializer=tf.truncated_normal_initializer())
            self.b2 = tf.get_variable("bias", shape=(layer_2_hidden), dtype=tf.float32, initializer=tf.constant_initializer(0.1))
            self.o2 = tf.nn.relu(tf.add(tf.matmul(self.o1,self.w2),self.b2), name="output")

        with tf.variable_scope("agent_{}".format(self.params['aid'])):
            self.w3 = tf.get_variable("weight", shape=(layer_2_hidden, self.params['num_act']), dtype=tf.float32, initializer=tf.truncated_normal_initializer())
            self.b3 = tf.get_variable("bias", shape=(self.params['num_act']), dtype=tf.float32, initializer=tf.constant_initializer(0.1))
            self.y = tf.add(tf.matmul(self.o2,self.w3),self.b3, name="output_aka_y")

        #Q,Cost,Optimizer
        with tf.variable_scope("optimizer"):
            self.discount = tf.constant(self.params['discount'], name="discount")
            self.yj = tf.add(self.rewards, tf.mul(1.0-self.terminals, tf.mul(self.discount, self.q_t)), name="true_y")
            self.Q_pred = tf.reduce_sum(tf.mul(self.y,self.actions), reduction_indices=1, name="q_pred")
            self.cost = tf.reduce_sum(tf.pow(tf.sub(self.yj, self.Q_pred), 2), name="cost")
            
            self.rmsprop_min = tf.train.RMSPropOptimizer(learning_rate=self.params['lr'],
                                                        decay=self.params['rms_decay'],
                                                        momentum=self.params['rms_momentum'],
                                                        epsilon=self.params['rms_eps']).minimize(self.cost)
    
    def set_session(self, session, global_step_var):
        self.sess = session
        self.global_step_var = global_step_var 
    
    def train(self, states, actions, rewards, terminals, next_states, allow_update=True):
        q_target_max = np.amax(self.q(next_states), axis=1) # Pick the next state's best value to use in the reward (curRew + discount*(nextRew))

        feed_dict={self.x: self.scale_state_input(states), self.q_t: q_target_max, self.actions: actions, self.rewards: rewards, self.terminals:terminals}
        step, _, costs = self.sess.run([self.global_step_var, self.rmsprop_min, self.cost], feed_dict=feed_dict)

        return costs, step
        
    def q(self, states):
        return self.sess.run(self.y, feed_dict={self.x: self.scale_state_input(states)})
    
    def scale_state_input(self, state_to_scale): # TODO move this to be part of tensorflow to speed things up
        if self.params['input_scaling_vector'] is None:
            return state_to_scale
        else:
            return np.array(state_to_scale) / self.params['input_scaling_vector']
    
    def save_weights(self, name, boolean_for_something):
        # SAVE WEIGHTS?!?  CLIENTS DON'T GET TO SAVE WEIGHTS.  THOSE COME FROM THE SERVER!
        pass
        