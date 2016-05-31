import time
import numpy as np
import client
import GenericAgent
import JacobsMazeWorld
import statistics
from networks import NetworkType, Messages




### COMMAND LINE ARGUMENTS ###
import argparse
parser = argparse.ArgumentParser()
# SHARED
parser.add_argument('--world_id', '-wid', default=1, type=int, help="ID of the world(maze) you want to use")
parser.add_argument('--task_id', '-tid',  default=1, type=int, help="ID of the task(start/end positions) you want to use")
parser.add_argument('--agent_id', '-aid', default=1, type=int, help="ID of the agent you want to use (nsew/sewn/ewns/etc")
# WORLD
parser.add_argument('--random_starting_location', '-rand_start', default=False, action='store_true')
# AGENT
parser.add_argument('--num_steps', '-ns', default=750000,  type=int)
parser.add_argument('--annealing_size', '-an', default=1500,  type=int)
parser.add_argument('--start_epsilon', default=1.0, type=float)
parser.add_argument('--end_epsilon', '-e', default=0.01,  type=float)
parser.add_argument('--boltzman_softmax', '-sm', default=False, action='store_true')
parser.add_argument('--observer', '-o', default=False, action='store_true')
parser.add_argument('--use_experience_replay', '-exp', default=False, action='store_true')
parser.add_argument('--ignore_evaluation_periods', '-no_eval', default=False, action='store_true')
parser.add_argument('--eval_episodes_between_evaluation', '-eval_steps', default=145, type=int, 
                     help="Ignored if evaluate_peridocally is false.  Run this many episodes before evaluating.  150 seems to be fine (TODO Verify)")
parser.add_argument('--eval_episodes_to_take', '-eval_len', default=15, type=int, 
                     help="Ignored if evaluate_peridocally is false.  defaults to (TODO find a good one)")
parser.add_argument('--codename', '-name', default="", type=str, help="code name used to display in sql")
parser.add_argument('--steps_til_train', '-stt', default=150, type=int)
# NEURAL-NET       #Discount Factor, Learning Rate, etc. TODO
parser.add_argument('--scale_input', default=False, action='store_true')
parser.add_argument('--discount_rate', '-disc', default=0.90, type=float)
parser.add_argument('--learning_rate', '-lr', default=0.0001, type=float)
parser.add_argument('--momentum', '-nnm', default=0.0, type=float) # 0 works well.
parser.add_argument('--allow_local_nn_weight_updates', '-nnu', default=False, action='store_true')
parser.add_argument('--requested_gpu_vram_percent', '-vram', default=0.02,  type=float)
parser.add_argument('--device_to_use', '-device', default=1,  type=int)
# RUNNER
parser.add_argument('--max_steps_per_episode', '-msteps', default=150,  type=int)
parser.add_argument('--verbose', '-v', default=0,  type=int)
parser.add_argument('--report_to_sql', '-sql', default=False, action='store_true')
parser.add_argument('--uuddlrlrba', '-udud', default=False, action='store_true')
# CLIENT-SERVER
parser.add_argument('--gradients_until_send', '-grads', default=1,  type=int)
parser.add_argument('--ignore_server', '-is', default=False, action='store_true')
parser.add_argument('--num_parallel_learners', '-npar', default=-1, type=int)
args = parser.parse_args()
### COMMAND LINE ARGUMENTS ###

if args.observer:
    args.ignore_evaluation_periods = False
    args.eval_episodes_between_evaluation = 10
    args.eval_episodes_to_take = 10
    args.allow_local_nn_weight_updates = True
    args.epsilon = 0.08 # give him a little bit of random to get out of bad policies
    args.annealing_size = 1
    args.gradients_until_send = 5
if args.uuddlrlrba:
    args.start_epsilon = 0.5
else:
    args.start_epsilon = 1

### OTHER FUNCTIONS ###
def send_gradients():
    if args.verbose >= 2: print "     sending Gradients!!!!"
    s = time.time()
    grads = agent.get_gradients()
    tf_client.sendGradients(grads[0], NetworkType.World)
    tf_client.sendGradients(grads[1], NetworkType.Task)
    tf_client.sendGradients(grads[2], NetworkType.Agent)
    end = time.time()
    if args.verbose >= 3: print "   Time to send gradients was {} seconds".format(end-s)
def cb(network_type, network_id):
    print "CALLBACK: {} {}".format(network_type, network_id)
    ws = tf_client.requestNetworkWeights(network_type)
    agent.set_weights(ws, network_type)

def uuddlrlrba_start_konami_cheat(verbose=False):
    ''' Gets a set experience database of small worlds, allows network to train perfectly, quickly'''
    # Grab all the experiences possible...
    print "\n\n=========\n=========\n Entering the cheat mode.\n=========\n=========\n\n"
    states = world.get_all_possible_states()
    if verbose:
        print "States:\n", states
        world.render()
    exp = []
    for state in states:
        for act in world.get_action_space():
            next_state, reward, terminal = world.act(act, state[0], state[1])
            exp.append([state, act, next_state, reward, terminal])
            if verbose:
                x = exp[-1]
                print "Action: {} goes from {} to {}, r:{}, t:{}".format(x[1], x[0], x[2], x[3], x[4])
    exp = np.array(exp).T
    if verbose:
        for x in exp:
            print "{}".format(x)
    
    # Now just train for ever!
    total_trains_to_do = 3000
    between_server_sends = 100
    if args.ignore_server:
        cost = agent.train_everything(total_trains_to_do, exp[0].tolist(),exp[1].tolist(),exp[2].tolist(), exp[3].tolist(), exp[4].tolist(), grads=True)
    else:
        for _ in range(total_trains_to_do/between_server_sends):
            cost = agent.train_everything(between_server_sends, exp[0].tolist(),exp[1].tolist(),exp[2].tolist(), exp[3].tolist(), exp[4].tolist(), grads=True)
            interact_with_server(True)
    print "\n\n=========\n=========\n Leaving the cheat mode.\n=========\n=========\n\n"
### OTHER FUNCTIONS ###


### Helpers
def printToConsole(is_eval, update_cnt, avgScore, max_q, cost, winning):
    print "%s = stp:%6d:: Re:%5.1f, Max_Q:%7.3f, c:%9.4f, E:%4.3f, W?:%s %s" % \
        ("{}.{}.{}".format(args.world_id, args.task_id, args.agent_id),
        update_cnt, avgScore, max_q, cost, agent.calculate_epsilon(),
        str(winning), "EVAL" if is_eval else "")
        
def Save_SQL_Evaluation(episode, num_steps, score, max_q, min_q, cost, did_win):
    database.save_evaluation_episode(learner_uuid = learner_uuid, episode = episode,
        steps_in_episode = num_steps, total_reward = score, q_max = max_q,cost = cost,
        end_epsilon = agent.calculate_epsilon(), did_win = did_win, is_evaluation=True)
        
def Save_SQL_Training_Step(update_cnt, num_steps, score, max_q, min_q, cost, winning_cnt):
    database.save_training_steps(learner_uuid = learner_uuid, update_cnt = update_cnt,
        steps_in_training = num_steps, total_reward = world.get_score(), q_max = max_q,
        cost = cost, end_epsilon = agent.calculate_epsilon(), number_wins = winning_cnt, is_evaluation=False)
        
def interact_with_server(send_grads=True):
    if send_grads:
        if args.verbose >= 1:
            print "SENDING GRADIENTS (x3)"
        send_gradients()
    tf_client.poll_once() # calls the callback added above if weights available!
    
def resetVariables():
    global max_q, min_q, sum_q, winning_cnt, running_score
    agent.reset_exp_db() # go ahead and reset now.  Note:exp will span multiple games now.
    max_q, min_q, sum_q = -np.Infinity, np.Infinity, 0.0
    winning_cnt, running_score = 0, 0.0
    
    
### Helpers
def runWorldOneStep(max_q, min_q):
    cur_state = world.get_state()
    action, values = agent.select_action(np.array(cur_state))
    next_state, reward, terminal = world.act(action)
    agent.stash_new_exp(cur_state, action, reward, terminal, next_state)
    return reward, max(max_q, np.max(values)),  min(min_q, np.min(values))
### Helpers


### INITIALIZE OBJECTS ###
world = JacobsMazeWorld.JacobsMazeWorld(
    world_id = args.world_id,
    task_id  = args.task_id,
    agent_id = args.agent_id,
    random_start = args.random_starting_location)

tf_client = client.ModDNN_ZMQ_Client(
    world_id = args.world_id,
    task_id  = args.task_id,
    agent_id = args.agent_id)

agent = GenericAgent.Agent(
    state_size=world.get_state_space(),
    number_of_actions=len(world.get_action_space()),
    input_scaling_vector=world.get_state__maxes() if args.scale_input else None, # Default None
    start_epsilon=args.start_epsilon,
    end_epsilon=args.end_epsilon,
    batch_size=250,
    learning_rate = args.learning_rate,
    momentum = args.momentum,
    discount = args.discount_rate,
    boltzman_softmax= args.boltzman_softmax,
    use_experience_replay=args.use_experience_replay,
    annealing_size=int(args.annealing_size), # annealing_size=args.annealing_size,
    allow_local_nn_weight_updates = args.allow_local_nn_weight_updates,
    requested_gpu_vram_percent = args.requested_gpu_vram_percent,
    device_to_use = args.device_to_use,
    )

if args.report_to_sql:
    learner_uuid = statistics.get_new_uuid()
if not args.ignore_server:
    tf_client.setWeightsAvailableCallback(cb)
    #Request and set initial weights
    agent.set_weights(tf_client.requestNetworkWeights(NetworkType.World), NetworkType.World)
    agent.set_weights(tf_client.requestNetworkWeights(NetworkType.Task),  NetworkType.Task)
    agent.set_weights(tf_client.requestNetworkWeights(NetworkType.Agent), NetworkType.Agent)
    if args.report_to_sql:
        server_uuid = tf_client.request_server_uuid()
else:
    if args.report_to_sql:
        server_uuid = learner_uuid # Cheating... but that's ok.  ... We could do None! but I'm not sure how that breaks whenwe get to sql...

print "\n\nCodename: {}".format(args.codename)
if args.report_to_sql:
    print "============================================================="
    print "======== CLIENT using LEARNER-UUID: {} ========".format(learner_uuid)
    print "======== CLIENT using SERVER-UUID:  {} ========".format(server_uuid)
    print "============================================================="
# SQL
if args.report_to_sql:
    database = statistics.Statistics(host="aji.cs.byu.edu", port=5432, db="mod_dnn_research")
    database.log_game_settings(
        learner_uuid=learner_uuid,
        parallel_learning_session_uuid=server_uuid,
        world_id=args.world_id,
        task_id=args.task_id,
        agent_id=args.agent_id,
        max_episode_count=args.max_steps_per_episode,
        annealing_size=args.annealing_size,
        final_epsilon=args.end_epsilon,
        num_parallel_learners=args.num_parallel_learners,
        using_experience_replay=agent.is_using_experience_replay(),
        codename=args.codename)
### INITIALIZE OBJECTS ###

if args.uuddlrlrba:
    uuddlrlrba_start_konami_cheat()

### RUN !!!!!!!!!!!!! ###
def is_eval_episode(e):
    is_eval = None
    if args.ignore_evaluation_periods:
        is_eval = False # Just keep on learning
    else:
        period = args.eval_episodes_between_evaluation + args.eval_episodes_to_take
        is_eval = e % period >= args.eval_episodes_between_evaluation
    return is_eval


print "\n\n====\n Beginning regular training \n====\n\n"
step_cnt, update_cnt, eval_episode = 1, 1, 0
max_q, min_q, sum_q, cost = -np.Infinity, np.Infinity, 0.0, 0.0
winning_cnt, running_score = 0, 0.0
agent.reset_exp_db()
while step_cnt < args.num_steps:
    world.reset()
    if is_eval_episode(update_cnt+eval_episode):
        eval_episode += 1
        agent.set_evaluate_flag(True)
        tmp_exp = agent.get_exp_db() # Save state
        resetVariables()
        while world.is_running() and world.get_time() < args.max_steps_per_episode:
            reward, max_q, min_q = runWorldOneStep(max_q, min_q)
            running_score += reward
        # Test your network & Report
        cost = agent.train(False)
        if args.verbose >= 0:
            printToConsole(True, eval_episode, world.get_score(), max_q, cost,
                           "Y ({:3})".format(world.get_time()) if not world.is_running() else "n (---)")
        if args.report_to_sql:
            Save_SQL_Evaluation(eval_episode, world.get_time(), world.get_score(), max_q, min_q, cost, world.is_running())
        agent.set_exp_db(tmp_exp)  # Restore state
    else:
        agent.set_evaluate_flag(False)
        while world.is_running() and world.get_time() < args.max_steps_per_episode:
            reward, max_q, min_q = runWorldOneStep(max_q, min_q)
            running_score += reward
            step_cnt += 1
            if step_cnt % args.steps_til_train == 0:
                update_cnt += 1
                cost = agent.train()
                if args.verbose >= 0:
                    printToConsole(False, update_cnt, running_score, max_q, cost, winning_cnt)
                if args.report_to_sql:
                    Save_SQL_Training_Step(update_cnt, args.steps_til_train, running_score, max_q, min_q, cost, winning_cnt)
                if not args.ignore_server:
                    interact_with_server(update_cnt % args.gradients_until_send == 0)
                resetVariables()

        
        if world.get_time() != args.max_steps_per_episode:
            winning_cnt += 1
