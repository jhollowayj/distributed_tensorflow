import sys
import uuid
import psycopg2
import psycopg2.extras

class Statistics():
    def __init__(self, host, port, db):
	try:
            print "Connecting to db..."
            self.conn = psycopg2.connect(host=host, port=port, database=db, user="pccl_users_w_underscores", password="beware_the_pccl")
            print "Connected."
            self.cur = self.conn.cursor()
	except:
            print "Unexpected error:", sys.exc_info() # [0]
            print "Not connected"

    def log_game_settings(self, learner_uuid, parallel_learning_session_uuid,
                    world_id, task_id, agent_id, max_episode_count,
                    annealing_size, final_epsilon, num_parallel_learners,
                    using_experience_replay, codename):
        try:
            self.cur.execute("INSERT INTO tf_united_game_settings (learner_uuid, \
                parallel_learning_session_uuid, world_id, task_id, agent_id, \
                max_episode_count, annealing_size, final_epsilon, num_parallel_learners, \
                using_experience_replay,codename) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)",
                    (psycopg2.extras.UUID_adapter(learner_uuid), 
                    psycopg2.extras.UUID_adapter(parallel_learning_session_uuid),
                    world_id, task_id, agent_id, max_episode_count,
                    annealing_size, float(final_epsilon), num_parallel_learners,
                    using_experience_replay, codename))
            self.conn.commit()
        except Exception as e:
            print "\n\n =============== ERROR: couldnt do log_tf_united_game_settings"
            print "Unexpected error:", sys.exc_info()[0]
            print e
            print 
            pass
             
    def save_episode(self, learner_uuid, episode, steps_in_episode, total_reward,
                    q_max, q_min, avg_action_value_n, avg_action_value_e,
                    avg_action_value_s, avg_action_value_w, mean_cost, 
                    end_epsilon, did_win, is_evaluation):
                    
        try:
            self.cur.execute("INSERT INTO tf_united_episode_stats (learner_uuid, \
                episode, steps_in_episode, total_reward, q_max, q_min, avg_action_value_n, \
                avg_action_value_e, avg_action_value_s, avg_action_value_w, mean_cost, \
                end_epsilon, did_win, is_evaluation) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)",
                    (psycopg2.extras.UUID_adapter(learner_uuid), episode, steps_in_episode,
                    total_reward, float(q_max), float(q_min), 
                    avg_action_value_n, avg_action_value_e, avg_action_value_s, avg_action_value_w,
                    mean_cost,  end_epsilon, did_win, is_evaluation))
            self.conn.commit()
        except Exception as e:
            print "\n\n =============== ERROR: couldnt do save_episode"
            print "Unexpected error:", sys.exc_info() # [0]
            print e
            print 
            pass
             
def get_new_uuid():
    return uuid.uuid4() # make a random UUID  (uuid1 should work too...)