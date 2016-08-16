import numpy as np
import tensorflow as tf
import time

class Vividict(dict):
    def __missing__(self, key):
        value = self[key] = type(self)()
        return value

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
        with tf.name_scope("placeholders"):
            self.x = tf.placeholder(tf.float32,[None, self.params['input_dims']], name="nn_x")
            self.q_t = tf.placeholder(tf.float32,[None], name="nn_q_t")

            self.actions = tf.placeholder(tf.float32, [None, self.params['num_act']], name="nn_actions")
            self.rewards = tf.placeholder(tf.float32, [None], name="nn_rewards")
            self.terminals = tf.placeholder(tf.float32, [None], name="nn_terminals")
        with tf.name_scope("Shared_stuff"):
            discount = tf.constant(self.params['discount'], name="discount") # only need one constant.  :)
            self.yj = tf.add(self.rewards, tf.mul(1.0-self.terminals, tf.mul(discount, self.q_t)), name="true_y")

        ### Network ###
        layer_1_hidden = self.params['layer_1_hidden']
        layer_2_hidden = self.params['layer_2_hidden']

        w1s, b1s, w2s, b2s, w3s, b3s = {}, {}, {}, {}, {}, {}
        self.ys, self.Q_preds, self.costs, self.rmsprop_mins = Vividict(), Vividict(), Vividict(), Vividict()
        for i in range(1,4):
            with tf.device("/job:ps/task:0"):
                with tf.variable_scope("world_{}".format(i)):
                    w1s[i] = tf.get_variable("weight", shape=(self.params['input_dims'], layer_1_hidden), dtype=tf.float32, initializer=tf.truncated_normal_initializer())
                    b1s[i] = tf.get_variable("bias", shape=(layer_1_hidden), dtype=tf.float32, initializer=tf.constant_initializer(0.1))
            
            with tf.device("/job:ps/task:1"):
                with tf.variable_scope("task_{}".format(i)):
                    w2s[i] = tf.get_variable("weight", shape=(layer_1_hidden, layer_2_hidden), dtype=tf.float32, initializer=tf.truncated_normal_initializer())
                    b2s[i] = tf.get_variable("bias", shape=(layer_2_hidden), dtype=tf.float32, initializer=tf.constant_initializer(0.1))
            
            with tf.device("/job:ps/task:2"):
                with tf.variable_scope("agent_{}".format(i)):
                    w3s[i] = tf.get_variable("weight", shape=(layer_2_hidden, self.params['num_act']), dtype=tf.float32, initializer=tf.truncated_normal_initializer())
                    b3s[i] = tf.get_variable("bias", shape=(self.params['num_act']), dtype=tf.float32, initializer=tf.constant_initializer(0.1))
        
        # Build the network here!
        for w in range(1,4): # Yuck... :( but it works.  We have to initiallize *all* of the variables on all of the worker nodes
            for t in range(1,4):  # So that the cheif worker (who is the only one authorized to do the actual initializing) can do
                for a in range(1,4):  # What he needs to do.  Then, once that's done, each worker can grab what he needs.  (see build_worker_specific_variables below)
                    # WORLD
                    with tf.variable_scope("world_{}".format(w)):
                        o1 = tf.nn.relu(tf.add(tf.matmul(self.x,w1s[w]),b1s[w]), name="output")
                    # TASK
                    with tf.variable_scope("task_{}".format(t)):
                        o2 = tf.nn.relu(tf.add(tf.matmul(o1,w2s[t]),b2s[t]), name="output")
                    # AGENT
                    with tf.variable_scope("agent_{}".format(a)):
                        self.ys[w][t][a] = tf.add(tf.matmul(o2,w3s[a]),b3s[a], name="output_y")

                    #Q,Cost,Optimizer for that WTA set
                    with tf.variable_scope("optimizer_{}.{}.{}".format(w,t,a)):
                        self.Q_preds[w][t][a] = tf.reduce_sum(tf.mul(self.ys[w][t][a],self.actions), reduction_indices=1, name="q_pred")
                        self.costs[w][t][a] = tf.reduce_sum(tf.pow(tf.sub(self.yj, self.Q_preds[w][t][a]), 2), name="cost")

                        self.rmsprop_mins[w][t][a] = tf.train.RMSPropOptimizer(
                            learning_rate=self.params['lr'],
                            decay=self.params['rms_decay'],
                            momentum=self.params['rms_momentum'],
                            epsilon=self.params['rms_eps']).minimize(self.costs[w][t][a])                        
    
    def build_worker_specific_variables(self):
        self.rmsprop_min = self.rmsprop_mins[self.params['wid']][self.params['tid']][self.params['aid']]
        self.cost = self.costs[self.params['wid']][self.params['tid']][self.params['aid']]
        self.y = self.ys[self.params['wid']][self.params['tid']][self.params['aid']]

    def set_session(self, session, global_step_inc):
        self.sess = session
        self.global_step_inc = global_step_inc 
    
    def train(self, states, actions, rewards, terminals, next_states, allow_update=True):
        q_target_max = np.amax(self.q(next_states), axis=1) # Pick the next state's best value to use in the reward (curRew + discount*(nextRew))

        feed_dict={self.x: self.scale_state_input(states), self.q_t: q_target_max, self.actions: actions, self.rewards: rewards, self.terminals:terminals}
        step, _, costs = self.sess.run([self.global_step_inc, self.rmsprop_min, self.cost], feed_dict=feed_dict)

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
        