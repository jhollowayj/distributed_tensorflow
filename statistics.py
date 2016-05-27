import sys
import uuid
import psycopg2
import psycopg2.extras


def get_new_uuid():
    return uuid.uuid4() # make a random UUID  (uuid1 should work too...)


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

###################################################################################################
###################################################################################################
    def log_game_settings(self, learner_uuid, parallel_learning_session_uuid,
                    world_id, task_id, agent_id, max_episode_count,
                    annealing_size, final_epsilon, num_parallel_learners,
                    using_experience_replay, codename, old_table_name = False):
        try:
            self.cur.execute("INSERT INTO {} (learner_uuid, \
                parallel_learning_session_uuid, world_id, task_id, agent_id, \
                max_episode_count, annealing_size, final_epsilon, num_parallel_learners, \
                using_experience_replay,codename) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)".format(
                    "tf_united_game_settings" if old_table_name else "game_settings"),
                    (psycopg2.extras.UUID_adapter(learner_uuid), 
                    psycopg2.extras.UUID_adapter(parallel_learning_session_uuid),
                    world_id, task_id, agent_id, max_episode_count,
                    annealing_size, float(final_epsilon), num_parallel_learners,
                    using_experience_replay, codename))
            self.conn.commit()
        except Exception as e:
            print "\n\n =============== ERROR: couldnt do log_game_settings"
            print "Unexpected error:", sys.exc_info()[0]
            print e
            print 
            pass
            
###################################################################################################
###################################################################################################

    def save_evaluation_episode(self, learner_uuid, episode, steps_in_episode, total_reward, 
                                q_max, cost, end_epsilon, did_win, is_evaluation=True):
        try:
            self.cur.execute("INSERT INTO evaluation_episode_values (learner_uuid, episode, \
                steps_in_episode, total_reward, q_max, cost, end_epsilon, did_win, is_evaluation) \
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)",
                    (psycopg2.extras.UUID_adapter(learner_uuid), episode, steps_in_episode,
                     float(total_reward), float(q_max), float(cost), end_epsilon, did_win, is_evaluation))
            self.conn.commit()
        except Exception as e:
            print "\n\n =============== ERROR: couldnt do save_evaluation_episode"
            print "Unexpected error:", sys.exc_info() # [0]
            print e
            print 
            pass
    def save_training_steps(self, learner_uuid, update_cnt, steps_in_training, total_reward,
                            q_max, cost, end_epsilon, number_wins, is_evaluation=False):
        try:
            self.cur.execute("INSERT INTO training_step_values (learner_uuid, \
                update_cnt, steps_in_training, total_reward, q_max, cost, \
                end_epsilon, number_wins, is_evaluation) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)",
                    (psycopg2.extras.UUID_adapter(learner_uuid), update_cnt, steps_in_training,
                     float(total_reward), float(q_max), float(cost), end_epsilon, number_wins, is_evaluation))
            self.conn.commit()
        except Exception as e:
            print "\n\n =============== ERROR: couldnt do save_training_steps"
            print "Unexpected error:", sys.exc_info() # [0]
            print e
            print 
            pass
            
###################################################################################################

    def save_episode(self, learner_uuid, episode, steps_in_episode, total_reward,
                     q_max, q_min, avg_action_value_n, avg_action_value_e,
                     avg_action_value_s, avg_action_value_w, mean_cost, 
                     end_epsilon, did_win, is_evaluation):
        try:
            self.cur.execute("INSERT INTO tf_united_episode_stats (learner_uuid, \
                episode, steps_in_episode, total_reward, q_max, q_min, mean_cost, \
                end_epsilon, did_win, is_evaluation) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)",
                    (psycopg2.extras.UUID_adapter(learner_uuid), episode, steps_in_episode,
                    total_reward, float(q_max), float(q_min), 
                    avg_action_value_n, avg_action_value_e, avg_action_value_s, avg_action_value_w,
                    mean_cost, end_epsilon, did_win, is_evaluation))
            self.conn.commit()
        except Exception as e:
            print "\n\n =============== ERROR: couldnt do save_episode"
            print "Unexpected error:", sys.exc_info() # [0]
            print e
            print 
            pass
