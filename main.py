import tensorflow as tf
import JacobsMazeWorld
from DQN import DQN
import GenericAgent
import time
import sys
import numpy as np

######################################################################################
#Flags for defining the tf.train.ClusterSpec
tf.app.flags.DEFINE_string("ps_hosts", "",
                           "Comma-separated list of hostname:port pairs")
tf.app.flags.DEFINE_string("worker_hosts", "",
                           "Comma-separated list of hostname:port pairs")

#Flags for defining the tf.train.Server
tf.app.flags.DEFINE_string("job_name", "", "One of 'ps', 'worker'")
tf.app.flags.DEFINE_integer("task_index", 0, "Index of task within the job")

#Flags for World, Task, Agent
tf.app.flags.DEFINE_integer("world_id", 1, "Index of world")
tf.app.flags.DEFINE_integer("task_id",  1, "Index of task") # don't confuse this with the distributed TF task_index
tf.app.flags.DEFINE_integer("agent_id", 1, "Index of agent")

tf.app.flags.DEFINE_boolean('random_starting_location', False, "If the world should start at a random location in maze")
tf.app.flags.DEFINE_boolean('state_as_xy', False, "Reduces the state representation to a xy instead of onehot state")
# AGENT
tf.app.flags.DEFINE_integer('num_steps', 750000, "number of steps to take before hard-stopping the program (a max)")
tf.app.flags.DEFINE_integer('annealing_size', 1500,  "exploration policy: Steps to go before we are reduced to greedy exploitation")
tf.app.flags.DEFINE_float  ('start_epsilon', 1.0, " exploration policy: Starting epsilon")
tf.app.flags.DEFINE_float  ('end_epsilon', 0.05,  "exploration policy: end epsilon")
tf.app.flags.DEFINE_boolean('boltzman_softmax', False, "Cant remember.  I think it's for selecting the action to take...")
tf.app.flags.DEFINE_boolean('observer', False, "Wheither to update gradients or not.  NOT CURRENTLY IMPLEMENTED")
tf.app.flags.DEFINE_boolean('use_experience_replay', False, "If true: keeps experience replay DB of size [memory].  False: simply replays that one game")
tf.app.flags.DEFINE_boolean('ignore_evaluation_periods', False, "Skips eval sessions if true.  usefull for observers")
tf.app.flags.DEFINE_integer('eval_episodes_between_evaluation', 145, "Ignored if evaluate_peridocally is false.  Run this many episodes before evaluating.  150 seems to be fine (TODO Verify)")
tf.app.flags.DEFINE_integer('eval_episodes_to_take', 15, "Ignored if evaluate_peridocally is false.  defaults to (TODO find a good one)")
tf.app.flags.DEFINE_string( 'codename', "", "code name used to display in sql")
tf.app.flags.DEFINE_integer('steps_til_train', 150, "Kinda like a burn in period I think")
tf.app.flags.DEFINE_integer('batch_size', 250, "How big of a batch to pull for the experience replay to use.  ignored if [use_experience_replay] is false")
# NEURAL-NET       #Discount Factor, Learning Rate, etc. TODO
tf.app.flags.DEFINE_boolean('scale_input', False, "Scales the input to be between 0-1")
tf.app.flags.DEFINE_float  ('discount_rate', 0.90, "Discount rate used in learner")
tf.app.flags.DEFINE_float  ('learning_rate', 0.0001, "Learniing rate used in learner")
tf.app.flags.DEFINE_float  ('momentum', 0.0, "Momentum used in learner") # 0 works well.
tf.app.flags.DEFINE_float  ('requested_gpu_vram_percent', 0.02, "How much gpu vram to use (DistTF doesn't support it yet for some reason with 'sv.prepare_or_wait_for_session')")
tf.app.flags.DEFINE_integer('device_to_use', 1, "Which gpu device to use.  Probably 0 if using 'cuda_visible_devices=#' before the python command")
# RUNNER
tf.app.flags.DEFINE_integer('max_steps_per_episode', 150, "Number of steps the game can try before it's declared 'game over'")
tf.app.flags.DEFINE_integer('verbose', 0, "Level of prints to use (0=none, 1, 2, 3)")
tf.app.flags.DEFINE_boolean('report_to_sql', False, "Send numbers to sql.  Defaults to false.")
# CLIENT-SERVER
tf.app.flags.DEFINE_integer('num_parallel_learners', -1, "Mostly just used for sql logging.")
FLAGS = tf.app.flags.FLAGS

if FLAGS.observer:
    FLAGS.ignore_evaluation_periods = False
    FLAGS.eval_episodes_between_evaluation = 10
    FLAGS.eval_episodes_to_take = 10
    FLAGS.start_epsilon = 0.08 # give him a little bit of random to get out of bad policies
    FLAGS.end_epsilon = 0.08   # give him a little bit of random to get out of bad policies
    FLAGS.annealing_size = 1

class Runner:
    def __init__(self, FLAGS):
      self.FLAGS = FLAGS
      
    def main(self):
        ps_hosts = self.FLAGS.ps_hosts.split(",")
        worker_hosts = self.FLAGS.worker_hosts.split(",")

        # Create a cluster from the parameter server and worker hosts.
        cluster = tf.train.ClusterSpec({"ps": ps_hosts, "worker": worker_hosts})

        # Create and start a server for the local task.
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.1, allow_growth=True)
        server = tf.train.Server(cluster,
                                job_name=self.FLAGS.job_name,
                                task_index=self.FLAGS.task_index)
                                # config=tf.ConfigProto(gpu_options=gpu_options)) Will be available in the next release
        self.build_classes()

          ##########################################################################################
        if self.FLAGS.job_name == "ps":
            server.join()
        elif self.FLAGS.job_name == "worker":
            # Build local graph/session
            with tf.Graph().as_default() as local_graph:
                self.dqn.build_worker_specific_model(FLAGS.task_index, local_graph)
                local_graph.finalize() # Just the local graph plz!

            # Then build the global stuff.
            tf.reset_default_graph()
            with tf.Graph().as_default() as global_graph:
                with tf.device(tf.train.replica_device_setter(worker_device="/job:worker/task:%d" % FLAGS.task_index, cluster=cluster)):
                    # Assign the variables to parameter servers, build all of the graphs
                    global_vars = self.dqn.build_global_variables()  
                    with tf.device("/job:ps/task:0") and tf.name_scope('global_vars'):
                            global_step_var = tf.Variable(0)
                            global_step_inc = global_step_var.assign_add(tf.constant(1))    
                ###################################################################################
                # Create a "supervisor", which oversees the training process.
                print "=============== building supervisor "
                sv = tf.train.Supervisor(is_chief=(self.FLAGS.task_index == 0),
                                        logdir="/mnt/pccfs/projects/distTF/modularDNN_Practice/logs/",
                                        init_op=tf.initialize_all_variables(),
                                        # summary_op=summary_op,
                                        # saver=saver,
                                        global_step=global_step_var)
                ###################################################################################

                start_time = time.time()
                print "=============== about to generate a sess (prepare_or_wait_for_session) "
                with sv.prepare_or_wait_for_session(server.target) as sess:
                
                    self.dqn.set_global_session(sess, global_step_inc, global_step_var) # Give him a session.
                    
                    print("\nSTARTING UP THE TRAINING STEPS =-=-=-=-=-=-=-=-=-=-=-=\n")
                    sys.stdout.flush()
        
                    step_cnt, update_cnt, eval_episode = 1, 1, 0
                    max_q, min_q, sum_q, winning_cnt, running_score = self.reset_variables()
                    gstep = 0
                    self.agent.reset_exp_db()
                    # Time to play the game vvv
                    print "About to start..."
                    sys.stdout.flush()
                    while not sv.should_stop() and step_cnt < self.FLAGS.num_steps:
                        # print "Global Step: {} || Local Step: {}".format(gstep, step_cnt)
                        # sys.stdout.flush()
                        
                        self.world.reset()
                        if self.is_eval_episode(update_cnt+eval_episode):
                            eval_episode += 1
                            self.agent.set_evaluate_flag(True)
                            tmp_exp = self.agent.get_exp_db() # Save state
                            max_q, min_q, sum_q, winning_cnt, running_score = self.reset_variables()
                            while self.world.is_running() and self.world.get_time() < self.FLAGS.max_steps_per_episode:
                                reward, max_q, min_q = self.run_world_one_step(max_q, min_q)
                                running_score += reward
                            # Test your network & Report
                            cost, gstep = self.agent.train(False)
                            self.agent.set_exp_db(tmp_exp)  # Restore state
                        else:
                            self.agent.set_evaluate_flag(False)
                            while self.world.is_running() and self.world.get_time() < self.FLAGS.max_steps_per_episode:
                                reward, max_q, min_q = self.run_world_one_step(max_q, min_q)
                                running_score += reward
                                step_cnt += 1
                                if step_cnt % self.FLAGS.steps_til_train == 0:
                                    update_cnt += 1
                                    cost, gstep = self.agent.train()
                                    max_q, min_q, sum_q, winning_cnt, running_score = self.reset_variables()
                                    self.print_to_console(False, update_cnt, running_score, max_q, cost, winning_cnt)
                                    
                        if self.world.get_time() != self.FLAGS.max_steps_per_episode:
                            winning_cnt += 1
            
                    # Once done, ask for all the services to stop.
                sv.stop()

    def build_classes(self):
        self.world = JacobsMazeWorld.JacobsMazeWorld(
            world_id = self.FLAGS.world_id,
            task_id  = self.FLAGS.task_id,
            agent_id = self.FLAGS.agent_id,
            random_start = self.FLAGS.random_starting_location,
            onehot_state = True)

        self.dqn = DQN(wid=self.FLAGS.world_id, tid=self.FLAGS.task_id, aid=self.FLAGS.agent_id,
            input_dims = self.world.get_state_space()[0],
            num_act = len(self.world.get_action_space()),
            input_scaling_vector=self.world.get_state__maxes() if self.FLAGS.scale_input else None, # Default None
            lr = self.FLAGS.learning_rate, 
            rms_momentum = self.FLAGS.momentum, 
            discount = self.FLAGS.discount_rate)

        self.agent = GenericAgent.Agent(dqn=self.dqn, 
            number_of_actions=len(self.world.get_action_space()), just_greedy=False,
            start_epsilon=self.FLAGS.start_epsilon,
            end_epsilon=self.FLAGS.end_epsilon,
            batch_size=self.FLAGS.batch_size, memory_size=10000, 
            boltzman_softmax= self.FLAGS.boltzman_softmax,
            use_experience_replay=self.FLAGS.use_experience_replay,
            annealing_size=int(self.FLAGS.annealing_size) )# annealing_size=self.FLAGS.annealing_size,

    ### Helpers
    def run_world_one_step(self, max_q, min_q):
        cur_state = self.world.get_state()
        action, values = self.agent.select_action(np.array(cur_state))
        next_state, reward, terminal = self.world.act(action)
        self.agent.stash_new_exp(cur_state, action, reward, terminal, next_state)
        return reward, max(max_q, np.max(values)),  min(min_q, np.min(values))
        
    def is_eval_episode(self, e):
        is_eval = None
        if self.FLAGS.ignore_evaluation_periods:
            is_eval = False # Just keep on learning
        else:
            period = self.FLAGS.eval_episodes_between_evaluation + self.FLAGS.eval_episodes_to_take
            is_eval = e % period >= self.FLAGS.eval_episodes_between_evaluation
        return is_eval
        
    def print_to_console(self, is_eval, update_cnt, avgScore, max_q, cost, winning):
        print "%s = stp:%6d:: Re:%5.1f, Max_Q:%7.3f, c:%9.4f, E:%4.3f, W?:%s %s" % \
        ("{}.{}.{}".format(self.FLAGS.world_id, self.FLAGS.task_id, self.FLAGS.agent_id),
        update_cnt, avgScore, max_q, cost, self.agent.calculate_epsilon(),
        str(winning), "EVAL" if is_eval else "")
        
    def reset_variables(self):
        self.agent.reset_exp_db() # go ahead and reset now.  Note:exp will span multiple games now.
        return -np.Infinity, np.Infinity, 0.0, 0, 0.0 # max_q, min_q, sum_q, winning_cnt, running_score
       

if __name__ == '__main__':
  r = Runner(FLAGS)
  r.main()