import random
import numpy as np
from DQN import DQN

class Agent:
    def __init__(self, state_size=None, number_of_actions=1, just_greedy=False,
                 epsilon=0.1, batch_size=200, discount=0.99, memory=10000, boltzman_softmax = False,
                 save_name='basic', save_freq=10, annealing_size=100, use_experience_replay=True,
                 input_scaling_vector=None, allow_local_nn_weight_updates=False,
                 requested_gpu_vram_percent = 0.01, device_to_use = 0):
        self.state_size = state_size
        self.number_of_actions = number_of_actions
        self.epsilon = epsilon
        self.batch_size = batch_size
        self.discount = discount
        self.save_name = save_name
        self.i = 1
        self.save_freq = save_freq
        self.iterations = 0
        self.boltzman_softmax = boltzman_softmax
        self.annealing_size = annealing_size # number of steps, not number of games 
        self.just_greedy = just_greedy
        self.input_scaling_vector = input_scaling_vector
        self.allow_local_nn_weight_updates = allow_local_nn_weight_updates
        self.requested_gpu_vram_percent = requested_gpu_vram_percent
        self.device_to_use = device_to_use
        self.is_eval = False
        
        self.use_exp_replay = use_experience_replay
        self.memory = memory 
        
        self.states = []
        self.actions = []
        self.rewards = []
        self.terminal = []
        self.next_state = []

        self.build_model()

        if self.allow_local_nn_weight_updates:
            print "NOTE: network is allowed to update his own weights!" 
        if self.input_scaling_vector is None:
            print "Not able to scale input.  This may lead to exploding gradients with multiple clients."
    
    def build_model(self):
        self.model = DQN(
            input_dims = self.state_size[0],
            num_act = self.number_of_actions,
            allow_local_nn_weight_updates = self.allow_local_nn_weight_updates,
            requested_gpu_vram_percent = self.requested_gpu_vram_percent,
            device_to_use = self.device_to_use)
        self.train_fn = self.model.train
        self.value_fn = self.model.q

    def new_episode(self):
        if self.use_exp_replay:
            self.states.append([])
            self.actions.append([])
            self.rewards.append([])
            self.terminal.append([])
            self.next_state.append([])
            
            self.states     = self.states[-self.memory:]
            self.actions    = self.actions[-self.memory:]
            self.rewards    = self.rewards[-self.memory:]
            self.terminal   = self.terminal[-self.memory:]
            self.next_state = self.next_state[-self.memory:]
        else:
            self.states = []
            self.actions = []
            self.rewards = []
            self.terminal = []
            self.next_state = []
        self.i += 1
        if self.i % self.save_freq == 0:
            self.model.save_weights('{}.h5'.format(self.save_name), True)
                      
    def end_episode(self):
        pass

    def is_using_experience_replay(self):
        return self.use_exp_replay

    def set_evaluate_flag(self, flag):
        self.is_eval = flag

    def stash_new_exp(self, cur_state, action, reward, terminal, next_state):
        if self.use_exp_replay:
            self.states[-1].append(cur_state)
            self.actions[-1].append(action)
            self.rewards[-1].append(reward)
            self.terminal[-1].append(terminal)
            self.next_state[-1].append(next_state)
        else:
            self.states.append(cur_state)
            self.actions.append(action)
            self.rewards.append(reward)
            self.terminal.append(terminal)
            self.next_state.append(next_state)

    def calculate_epsilon(self):
        epsilon = None
        if self.is_eval:
            epsilon = 0
        elif self.just_greedy:
            epsilon = self.epsilon
        else:
            epsilon = max(self.epsilon, 1-float(self.iterations) / self.annealing_size)
            # epsilon = max(self.epsilon, 1-float(self.iterations % (self.annealing_size*4.5)) / self.annealing_size)
        return epsilon

    def select_action(self, state):
        # # This use to scale the input, but we have to change how we do this.  Maybe pass it into the DQN instead, so it's always happening
        # if self.input_scaling_vector is not None:
        #     # Scale the input to be between 0 and 1.  # Supposed to help with exploding gradients
        #     if len(state) != len(self.input_scaling_vector): # Should probably be an assert instead
        #         print "+=============+ ERROR +==============+\n scaling input doesn't have the same shape... :("
        #     else:
        #         # tmp = state
        #         state = 1.0 * state / self.input_scaling_vector
        #         # print "State {} became {}".format(tmp, state) # Debugging
        values = self.value_fn([state])
        if np.random.random() < self.calculate_epsilon():
            if self.boltzman_softmax:
                Probz = values[0] + 0.1 - values[0].min()
                action = np.random.choice(self.number_of_actions, p=Probz/Probz.sum())
            else:
                action = np.random.randint(self.number_of_actions)
        else: 
            action = values.argmax()
        return action, values

    def train(self):
        self.iterations += 1
        if self.use_exp_replay:
            S, A, R, T, NS = self._calc_training_data__exp_rep()
        else:
            S, A, R, T, NS = self._calc_training_data_no_exp_rep()
        cost = self.train_fn(np.array(S), np.array(A), np.array(R), np.array(T), np.array(NS))
        return cost

    def _calc_training_data__exp_rep(self):
        N = len(self.states)

        # P = np.array([sum(episode)/len(episode) for episode in self.rewards]).astype(float)
        # P *= np.abs(P) # Square it (keeping the sign)
        # P -= np.min(P) # Set min to zero
        # P += 1            # still give the little guy a chance
        # P /= np.sum(P) # Scale to 0-1 (to sum to one)
        S, A, R, T, NS = [], [], [], [], []
        for i in xrange(self.batch_size):
            episodeId = random.randint(max(0, N-self.memory), N-1)   # Pick a past episode
            # episodeId = np.random.choice(len(P), p=P) # Select according to probablility

            num_frames = len(self.states[episodeId])       # Choose a random state from that episode 
            frame = random.randint(0, num_frames-1)        #   (cont)

            S.append(self.states[episodeId][frame])        # Adds state to batch
            A.append(self.onehot(self.actions[episodeId][frame])) # Add action
            R.append(self.rewards[episodeId][frame])       # Add reward
            T.append(self.terminal[episodeId][frame])      # Is it the termial of that episode?
            NS.append(self.next_state[episodeId][frame])   # Add next state
            
        return np.array(S), np.array(A), np.array(R), np.array(T), np.array(NS)

    def _calc_training_data_no_exp_rep(self):
        S = self.states
        A = [self.onehot(act) for act in self.actions]
        R = self.rewards
        T = self.terminal
        NS= self.next_state
        return S, A, R, T, NS

    def onehot(self, action):
        arr = [0] * self.number_of_actions
        arr[action] = 1
        return arr

        
    def set_weights(self, weights, network_type):
        self.model.set_weights(weights, network_type)

    def get_gradients(self):
        grs = self.model.get_and_clear_gradients()
        # Note this is not dynamic!  It's hard coded for 3 layers, each with 2 items (w, b)
        g1 = grs[0:2]
        g2 = grs[2:4]
        g3 = grs[4:6]
        return [g1,g2,g3]
        
    def getRewardsPerSquare(self, world):
        vals = [np.zeros((10,10))] * 4
        for x in range(10):
            for y in range(10):
                state = world.get_state(x+1, y+1)
                action_values = self.value_fn([state])
                print "returned qvals: {}{} = {}".format(x, y, action_values)
                for action_index, action_value in enumerate(action_values[0]):
                    # print "=== ind:{}, val: {}".format(action_index, action_value)
                    vals[action_index][x][y] = action_value
        return vals
    
    
###############################################################################
###############################################################################
###############################################################################
#####################  THIS IS ME CHEATING...  ################################
###############################################################################
###############################################################################
###############################################################################

    index_state_2_1 = None
    def train_everything(self, num_episodes, state, action, next_state, reward, terminal):
        S = np.array(state)
        A = np.array([self.onehot(act) for act in action])
        R = np.array(reward)
        T = np.array(terminal)
        NS= np.array(next_state)
        
        # if index_state_2_1 is None:
        #     for s in state:
        #         if s[0] == 2 and s[1] == 1:
        #             index_state_2_1 == state

        import time
        t, onek = time.time(), []
        for episode in xrange(num_episodes):    
            cost = self.train_fn(S, A, R, T, NS, episode % 100 == 0)
            if episode % 15 == 0:
                a, b = self.select_action(np.array([1,2]))
                print "{}\te{}\tcost:{}\t [1,2]: {} (4) {:15.10f}{:15.10f}{:15.10f}{:15.10f}  -- {}".format("testing", episode, cost, 
                        1+a, b[0][0], b[0][1], b[0][2], b[0][3], b[0].shape)
                if episode % 1000 == 0:
                    onek.append(time.time() - t)
        e = time.time()
        print "\n\nTime:{}".format(e - t)
        for i in range(len(onek)):
            print "    {} : {}".format(1000*i, onek[i])
