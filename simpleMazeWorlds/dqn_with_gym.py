import random
import numpy
from mrkulk_dqn import DQN

class Agent:
    def __init__(self, state_size=None, number_of_actions=1, just_greedy=False,
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
        self.build_model()
        self.iterations = 0
        self.anealing_size = anealing_size * 100.0 # Steps * percent
        self.just_greedy = just_greedy
        
    def build_model(self):
        self.model = DQN(
            input_dims = self.state_size[0],
            num_act = self.number_of_actions)
        self.train_fn = self.model.train
        self.value_fn = self.model.q

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
        if self.just_greedy:
            return self.epsilon
        else:
            repeat_random_periodically = False
            if repeat_random_periodically:
                iterationCnt = (self.iterations) % (self.anealing_size*4.5)
            else:
                iterationCnt = self.iterations
            return max(self.epsilon, 1-(iterationCnt / self.anealing_size))
    def select_action(self, state, append=True):
        if append:
            self.episodes[-1].append(state)
        arr = []
        arr.append(state)
        values = self.value_fn(arr)
        
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
        S, NS, A, R, T = [], [], [], [], []  
        
        P = numpy.array([sum(episode)/len(episode) for episode in self.rewards]).astype(float)
        P *= numpy.abs(P) # Square it (keeping the sign)
        P -= numpy.min(P)
        P += 1 # still give the little guy a chance
        P /= numpy.sum(P)
            
        randomEpisode = True
        for i in xrange(self.batch_size):
            if randomEpisode:
                episodeId = random.randint(max(0, N-self.memory), N-1)   # Pick a past episode
            else:
                episodeId = numpy.random.choice(len(P), p=P)
            # print "Chosens stuff: {}/{} L:{} R:{}".format(episodeId, len(self.episodes), len(self.episodes[episodeId]), self.rewards[episodeId][-1])
            
            num_frames = len(self.episodes[episodeId])      # Choose a random state from that episode 
            frame = random.randint(0, num_frames-1)         #   (cont)
            
            S.append(self.episodes[episodeId][frame])          # Adds state to batch
            T.append(1 if frame == num_frames - 1 else 0)      # Is it the termial of that episode?
            NS.append(self.episodes[episodeId][frame+1] if frame < num_frames - 1 else None)   #   Add next state
            A.append(self.onehot(self.actions[episodeId][frame]))           # Add action
            R.append(self.rewards[episodeId][frame])           # Add reward
            
            
        S, NS, A, R, T = numpy.array(S), numpy.array(NS), numpy.array(A), numpy.array(R), numpy.array(T)
        
        self.iterations += 1
        cost = self.train_fn(S, A, T, NS, R)              # TRAIN!!!!!!!
        
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
                
        