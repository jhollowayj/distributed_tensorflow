import numpy as np
import JacobsMazeWorld, GenericAgent, DQN


class Runner:
    def __init__(self, world_id=0, task_id=0, agent_id=0, DQN=None,
                 annealing_size=100, epsilon=0.1, just_observe=False,
                 num_episodes=3000, max_steps_per_episode=200,
                 write_csv=False, csv_filename="myFileName.csv", verbose=False,
                 print_tag=""):
                 
        if DQN is None: 
            print "ERROR, DQN is not given.  Can't play games now... :("
            return None # TODO Maybe we should throw an error here instead
             
        self.world = JacobsMazeWorld.JacobsMazeWorld( world_id=world_id, task_id=task_id, agent_id=agent_id)
        self.agent = GenericAgent.Agent(
            DQN = DQN,
            number_of_actions=len(self.world.get_action_space()),
            input_scaling_vector=self.world.get_state__maxes(),
            epsilon=epsilon,
            annealing_size=annealing_size,
        )
        
        self.params = {
            'csv_filename': csv_filename,
            'max_steps_per_episode': max_steps_per_episode,
            'num_episodes': num_episodes,
            'verbose': verbose,
            'write_csv': write_csv,
            'print_tag': print_tag,
        }

        self.episode_count = 0 
        if self.params['write_csv']:
            self.csv = open(self.params['csv_filename'],'w', 0)
            self.csv.write("episode,total_reward,cost,max_q,endEpsilon,didFinish\n")
       
    # def __del__(self): # note on __del__'s => http://www.andy-pearce.com/blog/posts/2013/Apr/python-destructor-drawbacks/
    #     self.close()   # clean up the resources
        
    def close(self):
        if self.params['write_csv']: self.csv.close()
    
    def step(self):
        '''Plays one full game, then trains the network'''
        self.episode_count += 1
        done = False
        self.world.reset()
        self.agent.new_episode()
        frame, max_q, min_q = 0, 0 - np.Infinity, np.Infinity
        act_Vals = [0,0,0,0] #HARD CODED
        while self.world.is_running() and self.world.get_time() < self.params['max_steps_per_episode']: 
            frame += 1 # Same as self.world.get_time()
            
            cur_state = self.world.get_state()
            action, values = self.agent.select_action(np.array(cur_state))
            reward = self.world.act(action)

            max_q  = max(max_q, np.max(values))
            min_q  = min(min_q, np.min(values))
            act_Vals += values # I think that works...
            
        cost = self.agent.train(reward) # Train at the end of the game!
        act_Vals = act_Vals[0]
        print "%s = episode: %6d:: Re:%5.1f, QMa/Mi/%7.3f/%7.3f,  avVal_NSEW:[%7.2f/ %7.2f/ %7.2f/ %7.2f], m.cost: %9.4f, end.E: %4.3f, W?: %s" % \
            (self.params['print_tag'], self.episode_count,  self.world.get_score(), max_q, min_q,
             act_Vals[0]/frame, act_Vals[1]/frame, act_Vals[2]/frame, act_Vals[3]/frame,
            (cost/frame), self.agent.calculate_epsilon(), "N" if self.world.is_running() else "Y")

        if self.params['write_csv']:
            self.csv.write("{},{},{},{},{},{}\n".format(e, self.world.get_score(), (cost/frame), max_q, self.agent.calculate_epsilon(), 0 if self.world.is_running() else 1))
        


global_DQN = DQN.DQN()
num_runners = 3
runners = []
num_episodes = 6000
runner_configs = [
    [0,0,0], [0,0,1], [0,0,2],
    [0,1,0], [0,1,1], [0,1,2],
    [0,2,0], [0,2,1], [0,2,2],
]
for i in range(num_runners):
    ids = runner_configs[i]
    runners.append(Runner(
        world_id=ids[0], task_id=ids[1], agent_id=ids[2], DQN=global_DQN,
        annealing_size=100, epsilon=1.0/20.0, just_observe=False,
        num_episodes=3000, max_steps_per_episode=200,
        write_csv=False, csv_filename="", verbose=False,
        print_tag="{}.{}.{}".format(ids[0],ids[1],ids[2]))
    )

# TODO breakout into multithread/multiprocess
for e in range(num_episodes): #Meh, why not?
    for r in runners:
        r.step()
    

