import numpy as np
import tensorflow as tf

class DQN:
    def __init__(self, input_dims = 2, num_act = 4,
            eps = 1.0, discount = 0.95, lr =  0.0002,
            rms_eps = 1e-6, rms_decay=0.99):
        self.params = {
            'input_dims': input_dims,
            'num_act': num_act,
            'discount': discount,
            'lr': lr,
            'rms_eps': rms_eps,
            'rms_decay': rms_decay
        }
        
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.333)
        self.sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
        
        self.x = tf.placeholder('float',[None, self.params['input_dims']], name="nn_x")
        self.q_t = tf.placeholder('float',[None], name="nn_q_t")

        self.actions = tf.placeholder("float", [None, self.params['num_act']], name="nn_actions")
        self.rewards = tf.placeholder("float", [None], name="nn_rewards")
        self.terminals = tf.placeholder("float", [None], name="nn_terminals")


        ### Network ###
        # Layer 1
        layer_1_hidden = 100
        self.w1 = tf.Variable(tf.random_normal([self.params['input_dims'], layer_1_hidden], stddev=0.01), name="nn_L1_w")
        self.b1 = tf.Variable(tf.constant(0.1, shape=[layer_1_hidden]), name="nn_L1_b")
        self.ip1 = tf.add(tf.matmul(self.x,self.w1),self.b1, name="nn_L1_ip")
        self.o1 = tf.nn.relu(self.ip1, name="nn_L1_output")
        # Layer 2        
        layer_2_hidden = 100
        self.w2 = tf.Variable(tf.random_normal([layer_1_hidden, layer_2_hidden], stddev=0.01), name="nn_L2_w")
        self.b2 = tf.Variable(tf.constant(0.1, shape=[layer_2_hidden]), name="nn_L2_b")
        self.ip2 = tf.add(tf.matmul(self.o1,self.w2),self.b2, name="nn_L2_ip")
        self.o2 = tf.nn.relu(self.ip2, name="nn_L2_output")
        # Last layer
        self.w3 = tf.Variable(tf.random_normal([layer_2_hidden, self.params['num_act']], stddev=0.01), name="nn_L3_w")
        self.b3 = tf.Variable(tf.constant(0.1, shape=[self.params['num_act']]), name="nn_L3_b")
        self.y = tf.add(tf.matmul(self.o2,self.w3),self.b3, name="y_OR_nn_L3_output")
        ### Gradients ###
        self._grad_w1 = np.zeros(([self.params['input_dims'], layer_1_hidden]))
        self._grad_b1 = np.zeros(([layer_1_hidden]))
        self._grad_w2 = np.zeros(([layer_1_hidden, layer_2_hidden]))
        self._grad_b2 = np.zeros(([layer_2_hidden]))
        self._grad_w3 = np.zeros(([layer_2_hidden, self.params['num_act']]))
        self._grad_b3 = np.zeros(([self.params['num_act']]))
        self._gradient_list = [ self._grad_w1, self._grad_b1, self._grad_w2, self._grad_b2, self._grad_w3, self._grad_b3]
        ### end Gradients ###
        
        #Q,Cost,Optimizer
        self.discount = tf.constant(self.params['discount'], name="nn_discount")
        self.yj = tf.add(self.rewards, tf.mul(1.0-self.terminals, tf.mul(self.discount, self.q_t)), name="nn_yj_whatever-that-is")
        self.Q_pred = tf.reduce_sum(tf.mul(self.y,self.actions), reduction_indices=1, name="nn_q_pred")
        self.cost = tf.reduce_sum(tf.pow(tf.sub(self.yj, self.Q_pred), 2), name="nn_cost")

        # self.rmsprop = tf.train.RMSPropOptimizer(self.params['lr'],self.params['rms_decay'],0.0,self.params['rms_eps']).minimize(self.cost) # Orig
        self.rmsprop = tf.train.RMSPropOptimizer(self.params['lr'],self.params['rms_decay'],0.0,self.params['rms_eps'])
        self.comp_grad = self.rmsprop.compute_gradients(self.cost)
        self.app_grads = self.rmsprop.apply_gradients(self.comp_grad)
        
        init = tf.initialize_all_variables()
        self.sess.run(init)
        print "###\n### Initialized MrKulk's network\n###"
    
    def set_weights(self, weights, network_type=1):
        todo = []
        if network_type == 1: # World
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
        for i, accumulate_grad in enumerate(self._gradient_list):
            accumulate_grad += episode_delta[i]

    def train(self,bat_s,bat_a,bat_t,bat_n,bat_r):
        # print bat_s,bat_a,bat_t,bat_n,bat_r
        feed_dict={self.x: bat_s, self.q_t: np.zeros(bat_n.shape[0]), self.actions: bat_a, self.terminals:bat_t, self.rewards: bat_r}
        q_t = self.sess.run(self.y,feed_dict=feed_dict)
        q_t = np.amax(q_t,axis=1)
        feed_dict={self.x: bat_s, self.q_t: q_t, self.actions: bat_a, self.terminals:bat_t, self.rewards: bat_r}
        
        grads = self.sess.run([grad for grad, _ in self.comp_grad][:6], feed_dict=feed_dict)
        self.stash_gradients(grads)
        _, cost = self.sess.run([self.app_grads, self.cost],feed_dict=feed_dict ) #really slow
        return cost

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
