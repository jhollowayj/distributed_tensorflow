import random
import numpy
from DQN import DQN

class Agent:
    def __init__(self, state_size=None, number_of_actions=1, just_greedy=False,
                 epsilon=0.1, batch_size=32, discount=0.99,
                 save_name='basic', save_freq=10, annealing_size=100,
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
        self.annealing_size = annealing_size * 100 # number of steps, not number of games 
        self.just_greedy = just_greedy
        self.input_scaling_vector = input_scaling_vector
        self.allow_local_nn_weight_updates = allow_local_nn_weight_updates
        self.requested_gpu_vram_percent = requested_gpu_vram_percent
        self.device_to_use = device_to_use

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
        self.states = []
        self.actions = []
        self.rewards = []
        self.i += 1
        if self.i % self.save_freq == 0:
            self.model.save_weights('{}.h5'.format(self.save_name), True)
            
    def end_episode(self):
        pass

    def calculate_epsilon(self):
        if self.just_greedy:
            return self.epsilon
        else:
            repeat_random_periodically = False
            if repeat_random_periodically:
                iterationCnt = (self.iterations) % (self.annealing_size*4.5)
            else:
                iterationCnt = self.iterations
            return max(self.epsilon, 1-(float(iterationCnt) / self.annealing_size))
    def select_action(self, state, append=True):
        if self.input_scaling_vector is not None:
            # Scale the input to be between 0 and 1.  # Supposed to help with exploding gradients
            if len(state) != len(self.input_scaling_vector):
                print "+=============+ ERROR +==============+\n scaling input doesn't have the same shape... :("
            else:
                # tmp = state
                state = 1.0 * state / self.input_scaling_vector
                # print "State {} became {}".format(tmp, state) # Debugging
            
        values = self.value_fn([state])
        
        # if numpy.random.random() < self.epsilon:
        if numpy.random.random() < self.calculate_epsilon():
            action = numpy.random.randint(self.number_of_actions)
        else: action = values.argmax()

        if append:
            self.states.append(state)
            self.actions.append(action)
            
        return action, values

    def train(self, reward):
        self.rewards.append(reward)
        return self.iterate()

    def iterate(self):
        self.iterations += 1

        S = self.states
        NS= S[1:] + [None]
        A = [self.onehot(act) for act in self.actions]
        R = self.rewards
        T = [0]*len(S)
        T[-1] = 1 # Last one was a terminal
        
        play_backwards = False
        if play_backwards:
            S.reverse()
            NS.reverse()
            A.reverse()
            R.reverse()
            T.reverse()
            T.reverse()
        
        cost = self.train_fn(
            numpy.array(S),
            numpy.array(A),
            numpy.array(T),
            numpy.array(NS),
            numpy.array(R))
        return cost

    def onehot(self, action):
        arr = [0] * self.number_of_actions
        arr[action] = 1
        return arr
        
    def set_weights(self, weights, network_type):
        self.model.set_weights(weights, network_type)

    def get_gradients(self):
        grs = self.model.get_and_clear_gradients()
        # Note this is not dynamic!  It's hard coded for 3 layers, each with 2 items (w, b) (I think it's right)
        g1 = grs[0:2]
        g2 = grs[2:4]
        g3 = grs[4:6]
        return [g1,g2,g3]
        
    def getRewardsPerSquare(self, world):
        vals = [numpy.zeros((10,10))] * 4
        for x in range(10):
            for y in range(10):
                state = world.get_state(x+1, y+1)
                action_values = self.value_fn([state])
                print "returned qvals: {}{} = {}".format(x, y, action_values)
                for action_index, action_value in enumerate(action_values[0]):
                    # print "=== ind:{}, val: {}".format(action_index, action_value)
                    vals[action_index][x][y] = action_value
        return vals
    