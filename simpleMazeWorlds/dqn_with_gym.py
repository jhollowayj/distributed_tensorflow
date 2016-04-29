import random
import numpy
from mrkulk_dqn import DQN

class Agent:
    def __init__(self, state_size=None, number_of_actions=1,
                 epsilon=0.1, batch_size=32, discount=0.99, memory=50,
                 save_name='basic', save_freq=10, anealing_size=100):
        self.state_size = state_size
        self.number_of_actions = number_of_actions
        self.epsilon = epsilon
        self.batch_size = batch_size
        self.discount = discount
        self.memory = memory # How many past episodes to remember (i.e. runs through the maze)
        self.save_name = save_name
        self.episodes = []
        self.actions = []
        self.rewards = []
        self.experience = []
        self.i = 1
        self.save_freq = save_freq
        self.build_functions()
        self.iterations = 0
        self.anealing_size = anealing_size * 100.0 # Steps * percent

    def build_model(self):
        v = "chris"
        S = Input(shape=self.state_size)
        h = Dense(1000, activation='relu')(S)
        h = Dense(1000, activation='relu')(h)
        h = Dense(1000, activation='relu')(h)
        V = Dense(self.number_of_actions)(h)

        self.model = Model(S, V)
        try:
            self.model.load_weights('{}.h5'.format(self.save_name))
            print "loading from {}.h5".format(self.save_name)
        except:
            print "Training a new model"

    def build_functions(self):
        S = Input(shape=self.state_size)
        NS = Input(shape=self.state_size)
        A = Input(shape=(1,), dtype='int32')
        R = Input(shape=(1,), dtype='float32')
        T = Input(shape=(1,), dtype='int32')
        self.build_model()
        self.value_fn = K.function([S], self.model(S))

        VS = self.model(S)
        VNS = disconnected_grad(self.model(NS))
        future_value = (1-T) * VNS.max(axis=1, keepdims=True)
        discounted_future_value = self.discount * future_value
        target = R + discounted_future_value
        cost = ((VS[:, A] - target)**2).mean()
        opt = RMSprop(0.0001)
        params = self.model.trainable_weights
        updates = opt.get_updates(params, [], cost)
        self.train_fn = K.function([S, NS, A, R, T], cost, updates=updates)

    def new_episode(self):
        self.episodes.append([])
        self.actions.append([])
        self.rewards.append([])
        self.episodes = self.episodes[-self.memory:]
        self.actions = self.actions[-self.memory:]
        self.rewards = self.rewards[-self.memory:]
        self.i += 1
        if self.i % self.save_freq == 0:
            self.model.save_weights('{}.h5'.format(self.save_name), True)

    def end_episode(self):
        pass

    def calculate_epsilon(self):
        iterationCnt = (self.iterations) % (self.anealing_size*4.5)
        return max(self.epsilon, 1-(iterationCnt / self.anealing_size))
    def select_action(self, state, append=True):
        if append:
            self.episodes[-1].append(state)
        values = self.value_fn([state[None, :]])
        
        # if numpy.random.random() < self.epsilon:
        if numpy.random.random() < self.calculate_epsilon():
            action = numpy.random.randint(self.number_of_actions)
        else:
            action = values.argmax()
        if append:
            self.actions[-1].append(action)
        return action, values

    def train(self, reward):
        self.rewards[-1].append(reward)
        return self.iterate()

    def iterate(self):
        N = len(self.episodes)
        S = numpy.zeros((self.batch_size,) + self.state_size)
        NS = numpy.zeros((self.batch_size,) + self.state_size)
        A = numpy.zeros((self.batch_size, 1), dtype=numpy.int32)
        R = numpy.zeros((self.batch_size, 1), dtype=numpy.float32)
        T = numpy.zeros((self.batch_size, 1), dtype=numpy.int32)
        
        P = numpy.array([sum(episode)/len(episode) for episode in self.rewards]).astype(float)
        P *= numpy.abs(P) # Square it (keeping the sign)
        P -= numpy.min(P)
        P += 1 # still give the little guy a chance
        P /= numpy.sum(P)
        # print "Probabili:", numpy.min(P), numpy.max(P), numpy.sum(P)
        # print "P = ", P 
        
            
        randomEpisode = True
        for i in xrange(self.batch_size):
            if randomEpisode:
                episodeId = random.randint(max(0, N-self.memory), N-1)   # Pick a past episode
            else:
                episodeId = numpy.random.choice(len(P), p=P)
            # print "Chosens stuff: {}/{} L:{} R:{}".format(episodeId, len(self.episodes), len(self.episodes[episodeId]), self.rewards[episodeId][-1])
            
            num_frames = len(self.episodes[episodeId])      # Choose a random state from that episode 
            frame = random.randint(0, num_frames-1)         #   (cont)
            
            S[i] = self.episodes[episodeId][frame]          # Adds state to batch
            T[i] = 1 if frame == num_frames - 1 else 0      # Is it the termial of that episode?
            if frame < num_frames - 1:                      # (If there is a next state)
                NS[i] = self.episodes[episodeId][frame+1]   #   Add next state
            A[i] = self.actions[episodeId][frame]           # Add action
            R[i] = self.rewards[episodeId][frame]           # Add reward
        cost = self.train_fn([S, NS, A, R, T])              # TRAIN!!!!!!!
        self.iterations += 1
        return cost


    def getRewardsPerSquare(self, indexer=0, maze=None):
        arr = numpy.zeros((12,12))
        for x in range(len(arr)):
            for y in range(len(arr[x])):
                state = numpy.array([x,y])
                res = self.value_fn([state[None, :]])
                arr[x][y] = int(res[0][indexer])
                # arr[x][y] = int(10*(res[0][0] - res[0][1]))+1
                # arr[x][y] = self.select_action(state, append=False)[0]
        arr *= maze
        return arr
                
        