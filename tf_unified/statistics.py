import uuid, sys, psycopg2, psycopg2.extras

class Statistics():
    def __init__(self, host, port, db):
	try:
            print "Connecting to db..."
            self.conn = psycopg2.connect(host=host, port=port, database=db, user="postgres", password="beware the pccl")
            print "Connected."
            self.cur = self.conn.cursor()
	except:
            print "Unexpected error:", sys.exc_info() # [0]
            print "Not connected"

    def log_tf_united_game_settings(self, learner_uuid, parallel_learning_session_uuid,
                    world_id, task_id, agent_id, max_episode_count,
                    annealing_size, final_epsilon, num_parallel_learners,
                    using_experience_replay):
        try:
            self.cur.execute("INSERT INTO tf_united_game_settings (learner_uuid, \
                parallel_learning_session_uuid, world_id, task_id, agent_id, \
                max_episode_count, annealing_size, final_epsilon, num_parallel_learners, \
                using_experience_replay) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)",
                    (psycopg2.extras.UUID_adapter(learner_uuid), 
                    psycopg2.extras.UUID_adapter(parallel_learning_session_uuid),
                    world_id, task_id, agent_id, max_episode_count,
                    annealing_size, float(final_epsilon), num_parallel_learners,
                    using_experience_replay))
            self.conn.commit()
        except:
            print "\n\n =============== ERROR: couldnt do log_tf_united_game_settings"
            print "Unexpected error:", sys.exc_info()[0]
            pass
             
    def save_episode(self, learner_uuid, episode, steps_in_episode, total_reward,
                    q_max, q_min, avg_action_value_n, avg_action_value_e,
                    avg_action_value_s, avg_action_value_w, mean_cost, 
                    end_epsilon, did_win):
                    
        try:
            self.cur.execute("INSERT INTO tf_united_episode_stats (learner_uuid, \
                episode, steps_in_episode, total_reward, q_max, q_min, avg_action_value_n, \
                avg_action_value_e, avg_action_value_s, avg_action_value_w, mean_cost, \
                end_epsilon, did_win) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)",
                    (psycopg2.extras.UUID_adapter(learner_uuid), episode, steps_in_episode,
                    total_reward, float(q_max), float(q_min), 
                    avg_action_value_n, avg_action_value_e, avg_action_value_s, avg_action_value_w,
                    mean_cost,  end_epsilon, did_win))
            self.conn.commit()
        except:
            print "\n\n =============== ERROR: couldnt do save_episode"
            print "Unexpected error:", sys.exc_info() # [0]
            pass
             
def get_new_uuid():
    return uuid.uuid4() # make a random UUID  (uuid1 works too...)