import statistics, JacobsMazeWorld, GenericAgent, numpy as np


class Runner:
    def __init__(self, world_id=0, task_id=0, agent_id=0, DQN=None,
                 annealing_size=100, epsilon=0.1, just_observe=False,
                 num_episodes=3000, max_steps_per_episode=200, verbose=True,
                 print_tag = "", 
                 parallel_learning_session_uuid = None,
                 num_parallel_learners=1):
                 
        if DQN is None: 
            print "ERROR, DQN is not given.  Can't play games now... :("
            return None # TODO Maybe we should throw an error here instead
        
        self.learner_uuid = statistics.get_new_uuid()
        self.database = statistics.Statistics(host="aji", port=5432, db="Jacob")
        self.episode_count = 0 
        
        self.world = JacobsMazeWorld.JacobsMazeWorld( world_id=world_id, task_id=task_id, agent_id=agent_id)
        self.agent = GenericAgent.Agent( DQN = DQN,
            number_of_actions=len(self.world.get_action_space()),
            input_scaling_vector=self.world.get_state__maxes(),
            epsilon=epsilon, annealing_size=annealing_size)
        self.params = {
            'max_steps_per_episode': max_steps_per_episode,
            'num_episodes': num_episodes,
            'verbose': verbose,
            'print_tag': print_tag,
        }
        
        self.database.log_tf_united_game_settings(
            learner_uuid = self.learner_uuid,
            parallel_learning_session_uuid = parallel_learning_session_uuid,
            world_id=world_id,
            task_id=task_id,
            agent_id=agent_id,
            max_episode_count=max_steps_per_episode,
            annealing_size=annealing_size, 
            final_epsilon=epsilon,
            num_parallel_learners=num_parallel_learners,
            using_experience_replay=self.agent.is_using_experience_replay())

        
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
        if self.params['verbose']:
            print "%s = episode: %6d:: Re:%5.1f, QMa/Mi/%7.3f/%7.3f,  avVal_NSEW:[%7.2f/ %7.2f/ %7.2f/ %7.2f], m.cost: %9.4f, end.E: %4.3f, W?: %s" % \
                (self.params['print_tag'], self.episode_count,  self.world.get_score(), max_q, min_q,
                act_Vals[0]/frame, act_Vals[1]/frame, act_Vals[2]/frame, act_Vals[3]/frame,
                (cost/frame), self.agent.calculate_epsilon(), "N" if self.world.is_running() else "Y")
        elif self.episode_count % 50 == 0:
            print "{} reached episode {}".format(self.params['print_tag'], self.episode_count)
            
        self.database.save_episode(
            learner_uuid = self.learner_uuid, episode = self.episode_count, steps_in_episode = frame,
            total_reward = self.world.get_score(), q_max = max_q, q_min = min_q, 
            avg_action_value_n = act_Vals[0]/frame, avg_action_value_e = act_Vals[1]/frame,
            avg_action_value_s = act_Vals[2]/frame, avg_action_value_w = act_Vals[3]/frame,
            mean_cost = cost/frame,  end_epsilon = self.agent.calculate_epsilon(),
            did_win = not self.world.is_running()
        )
