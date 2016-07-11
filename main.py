import tensorflow as tf
import JacobsMazeWorld
from DQN import DQN
import GenericAgent
import time

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
tf.app.flags.DEFINE_float(  'start_epsilon', 1.0, " exploration policy: Starting epsilon")
tf.app.flags.DEFINE_float(  'end_epsilon', 0.05,  "exploration policy: end epsilon")
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
tf.app.flags.DEFINE_float(  'discount_rate', 0.90, "Discount rate used in learner")
tf.app.flags.DEFINE_float(  'learning_rate', 0.0001, "Learniing rate used in learner")
tf.app.flags.DEFINE_float(  'momentum', 0.0, "Momentum used in learner") # 0 works well.
tf.app.flags.DEFINE_float(  'requested_gpu_vram_percent', 0.02, "How much gpu vram to use (DistTF doesn't support it yet for some reason with 'sv.prepare_or_wait_for_session')")
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

######################################################################################


def main(argv=None):
  ps_hosts = FLAGS.ps_hosts.split(",")
  worker_hosts = FLAGS.worker_hosts.split(",")

  # Create a cluster from the parameter server and worker hosts.
  cluster = tf.train.ClusterSpec({"ps": ps_hosts, "worker": worker_hosts})

  # Create and start a server for the local task.
  server = tf.train.Server(cluster,
                           job_name=FLAGS.job_name,
                           task_index=FLAGS.task_index)

  ##########################################################################################
  if FLAGS.job_name == "ps":
    server.join()
  elif FLAGS.job_name == "worker":

    # BUILD ALL THE CLASSES
    world = JacobsMazeWorld.JacobsMazeWorld(
          world_id = FLAGS.world_id,
          task_id  = FLAGS.task_id,
          agent_id = FLAGS.agent_id,
          random_start = FLAGS.random_starting_location,
          onehot_state = True)

    dqn = DQN(wid=FLAGS.world_id, tid=FLAGS.task_id, aid=FLAGS.agent_id,
          input_dims = world.get_state_space()[0],
          num_act = len(world.get_action_space()),
          input_scaling_vector=world.get_state__maxes() if FLAGS.scale_input else None, # Default None
          lr = FLAGS.learning_rate, 
          rms_momentum = FLAGS.momentum, 
          discount = FLAGS.discount_rate,
          requested_gpu_vram_percent = FLAGS.requested_gpu_vram_percent,
          device_to_use = FLAGS.device_to_use)
          
    agent = GenericAgent.Agent( dqn=dqn,
          start_epsilon=FLAGS.start_epsilon,
          end_epsilon=FLAGS.end_epsilon,
          batch_size=FLAGS.batch_size,
          boltzman_softmax= FLAGS.boltzman_softmax,
          use_experience_replay=FLAGS.use_experience_replay,
          annealing_size=int(FLAGS.annealing_size) )# annealing_size=args.annealing_size,

    with tf.name_scope('global_vars'):
        global_step_var = tf.Variable(0)

    # Run all the initializers to prepare the trainable parameters.
    saver = tf.train.Saver()                # dist
    summary_op = tf.merge_all_summaries()   # dist
    init_op = tf.initialize_all_variables() # dist
    
    ###################################################################################
    # Create a "supervisor", which oversees the training process.
    sv = tf.train.Supervisor(is_chief=(FLAGS.task_index == 0),
                             logdir="/mnt/pccfs/projects/distTF/mnist/logs/",
                             init_op=init_op,
                             summary_op=summary_op,
                             saver=saver,
                             global_step=global_step_var,
                             save_model_secs=600)
    ###################################################################################
    start_time = time.time()
    #with tf.Session() as sess:
    #with sv.managed_session(server.target) as sess:
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.1, allow_growth=True)
    print(gpu_options)
    with sv.prepare_or_wait_for_session(server.target, config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
      dqn.set_session(sess) # Give him a session.
      print("\nSTARTING UP THE TRAINING STEPS =-=-=-=-=-=-=-=-=-=-=-=\n")
      sys.stdout.flush()
      # Loop through training steps.
      step = 0
      while not sv.should_stop() and step < (int(num_epochs * train_size) // BATCH_SIZE):
        # Compute the offset of the current minibatch in the data.
        # Note that we could use better randomization across epochs.
        offset = (step * BATCH_SIZE) % (train_size - BATCH_SIZE)
        batch_data = train_data[offset:(offset + BATCH_SIZE), ...]
        batch_labels = train_labels[offset:(offset + BATCH_SIZE)]
        # This dictionary maps the batch data (as a numpy array) to the
        # node in the graph it should be fed to.
        feed_dict = {train_data_node: batch_data,
                     train_labels_node: batch_labels}
        # Run the graph and fetch some of the nodes.
        _, l, lr, predictions, globalstep = sess.run([optimizer, loss, learning_rate, train_prediction, batch], feed_dict=feed_dict)
        if step % EVAL_FREQUENCY == 0:
          elapsed_time = time.time() - start_time
          start_time = time.time()
          print('Step %d (%d) (epoch %.2f), %.1f ms' % (step, globalstep, float(step) * BATCH_SIZE / train_size, 1000 * elapsed_time / EVAL_FREQUENCY))
          print('Minibatch loss: %.3f, learning rate: %.6f' % (l, lr))
          print('Minibatch error: %.1f%%' % error_rate(predictions, batch_labels))
          print('Validation error: %.1f%%' % error_rate(eval_in_batches(validation_data, sess), validation_labels))
          print('')
          sys.stdout.flush()
      # Finally print the result!
      test_error = error_rate(eval_in_batches(test_data, sess), test_labels)
      print('Test error: %.1f%%' % test_error)
      if FLAGS.self_test:
        print('test_error', test_error)
        assert test_error == 0.0, 'expected 0.0 test_error, got %.2f' % (
            test_error,)
      sys.stdout.flush()
    # Ask for all the services to stop.
    sv.stop()


if __name__ == '__main__':
  tf.app.run()