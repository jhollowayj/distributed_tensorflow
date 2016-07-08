import random
import numpy as np

class Agent:
    def __init__(self, wid=1, tid=1, aid=1, dqn=None,
                 state_size=None, number_of_actions=1, just_greedy=False, start_epsilon=1.0,
                 end_epsilon=0.1, batch_size=200, memory_size=10000, boltzman_softmax = False,
                 save_name='basic', annealing_size=100, use_experience_replay=True,
                 input_scaling_vector=None, allow_local_nn_weight_updates=False,
                 learning_rate = 0.0002, momentum=0.0, discount = 0.90,
                 requested_gpu_vram_percent = 0.01, device_to_use = 0):
        self.world_id = wid
        self.task_id = tid
        self.agent_id = aid
        self.state_size = state_size
        self.number_of_actions = number_of_actions
        self.start_epsilon = start_epsilon
        self.end_epsilon = end_epsilon
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.discount = discount
        self.save_name = save_name
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
        self.memory_size = memory_size 
        
        self.states = []
        self.actions = []
        self.rewards = []
        self.terminal = []
        self.next_state = []
    
        self.model = dqn
        self.train_fn = self.model.train
        self.value_fn = self.model.q

    def reset_exp_db(self): # Was new_episode
        if self.use_exp_replay:
            self.states.append([])
            self.actions.append([])
            self.rewards.append([])
            self.terminal.append([])
            self.next_state.append([])
            
            self.states     = self.states[-self.memory_size:]
            self.actions    = self.actions[-self.memory_size:]
            self.rewards    = self.rewards[-self.memory_size:]
            self.terminal   = self.terminal[-self.memory_size:]
            self.next_state = self.next_state[-self.memory_size:]
        else:
            self.states = []
            self.actions = []
            self.rewards = []
            self.terminal = []
            self.next_state = []
    
    def get_exp_db(self):
        return [self.states, self.actions, self.rewards, self.terminal, self.next_state]
    
    def set_exp_db(self, new_db):
        self.states = new_db[0]
        self.actions = new_db[1]
        self.rewards = new_db[2]
        self.terminal = new_db[3]
        self.next_state = new_db[4]
        
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
            epsilon = 1.0/40.0 # it still needs a small random to break out of bad spots. 
        elif self.just_greedy:
            epsilon = self.end_epsilon
        else:
            # epsilon = max(self.epsilon, 1-float(self.iterations % (self.annealing_size*4.5)) / self.annealing_size)
            percent = 1 - min(1, max(0, float(self.iterations) / self.annealing_size)) # 0-1, linear
            range = self.start_epsilon - self.end_epsilon
            epsilon = self.end_epsilon + (percent * range)
        return epsilon

    def select_action(self, state):
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

    def train(self, allow_update=True):
        self.iterations += 1
                if self.use_exp_replay:
                S, A, R, T, NS = self._calc_training_data__exp_rep()
        else:
            S, A, R, T, NS = self._calc_training_data_no_exp_rep()
        cost = self.train_fn(np.array(S), np.array(A), np.array(R), np.array(T), np.array(NS), allow_update)
        return cost

    def _calc_training_data__exp_rep(self):
        N = len(self.states)

        S, A, R, T, NS = [], [], [], [], []
        for i in xrange(self.batch_size):

            # Pick a past episode
            episodeId = random.randint(max(0, N-self.memory_size), N-1)
            # Then a random frame from past episode
            num_frames = len(self.states[episodeId]) 
            frame = random.randint(0, num_frames-1)
            # Grab the experience
            S.append(self.states[episodeId][frame])
            A.append(self.onehot(self.actions[episodeId][frame]))
            R.append(self.rewards[episodeId][frame])
            T.append(self.terminal[episodeId][frame])
            NS.append(self.next_state[episodeId][frame])
            
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
