import time
import numpy as np
import AllKnowingAgent
import JacobsMazeWorld


### COMMAND LINE ARGUMENTS ###
import argparse
parser = argparse.ArgumentParser()
# SHARED
parser.add_argument('--world_id', '-wid', default=1, type=int, required=False, help="ID of the world(maze) you want to use")
parser.add_argument('--task_id', '-tid',  default=1, type=int, required=False, help="ID of the task(start/end positions) you want to use")
parser.add_argument('--agent_id', '-aid', default=1, type=int, required=False, help="ID of the agent you want to use (nsew/sewn/ewns/etc")
# AGENT
parser.add_argument('--num_episodes', '-ne', default=10000,  type=int, required=False, help="")
parser.add_argument('--annealing_size', '-an', default=3000,  type=int, required=False, help="")
parser.add_argument('--epsilon', '-e', default=0.04,  type=float, required=False, help="")
parser.add_argument('--observer', '-o', default=False,  action='store_true', required=False, help="")
parser.add_argument('--use_experience_replay', '-exp', default=False,  action='store_true', required=False, help="")
parser.add_argument('--evaluate_peridocally', '-eval', default=False, action='store_true', required=False, help="")
parser.add_argument('--eval_episodes_between_evaluation', '-eval_steps', default=145, type=int, required=False,
                    help="Ignored if evaluate_peridocally is false.  Run this many episodes before evaluating.  150 seems to be fine (TODO Verify)")
parser.add_argument('--eval_episodes_to_take', '-eval_len', default=5, type=int, required=False, 
                    help="Ignored if evaluate_peridocally is false.  defaults to (TODO find a good one)")
parser.add_argument('--codename', '-name', default="", type=str, required=False, help="code name used to display in sql")
# NEURAL-NET       #Discount Factor, Learning Rate, etc. TODO
parser.add_argument('--allow_local_nn_weight_updates', '-nnu', default=False,  action='store_true', required=False, help="")
parser.add_argument('--requested_gpu_vram_percent', '-vram', default=0.02,  type=float, required=False, help="")
parser.add_argument('--device_to_use', '-device', default=1,  type=int, required=False, help="")
# RUNNER
parser.add_argument('--max_steps_per_episode', '-msteps', default=200,  type=int, required=False, help="")
parser.add_argument('--verbose', '-v', default=0,  type=int, required=False, help="")
parser.add_argument('--report_to_sql', '-sql', default=False, action='store_true', required=False, help="")
args = parser.parse_args()
### COMMAND LINE ARGUMENTS ###

### INITIALIZE OBJECTS ###
world = JacobsMazeWorld.JacobsMazeWorld(
    world_id = args.world_id,
    task_id  = args.task_id,
    agent_id = args.agent_id)
    
agent = AllKnowingAgent.Agent(
    state_size=world.get_state_space(),
    number_of_actions=len(world.get_action_space()),
    input_scaling_vector=world.get_state__maxes(),
    epsilon=args.epsilon,
    batch_size=250,
    use_experience_replay=args.use_experience_replay,
    annealing_size=int(args.annealing_size), # annealing_size=args.annealing_size,
    allow_local_nn_weight_updates = args.allow_local_nn_weight_updates,
    requested_gpu_vram_percent = args.requested_gpu_vram_percent,
    device_to_use = args.device_to_use,
    )

### RUN !!!!!!!!!!!!! ###
def is_eval_episode(e):
    is_eval = None
    if args.evaluate_peridocally:
        period = args.eval_episodes_between_evaluation + args.eval_episodes_to_take
        is_eval = e % period < args.eval_episodes_between_evaluation
    else:
        is_eval = False # Ignore it by default
    return is_eval
    
         
update_cnt = 1
didwin, window = [], 25

# Grab all the experiences possible...
states = world.get_all_possible_states()
# print "States:\n", states
# world.render()
exp = []
for state in states:
    for act in world.get_action_space():
        next_state, reward, terminal = world.act(act, state[0], state[1])
        exp.append([state, act, next_state, np.clip(reward, -1, 1), terminal])
        # x = exp[-1]
        # print "Action: {} goes from {} to {}, r:{}, t:{}".format(x[1], x[0], x[2], x[3], x[4])
exp = np.array(exp).T
# for x in exp:
#     print "{}".format(x)

# Now just train for ever!
cost = agent.train_everything(args.num_episodes, exp[0].tolist(),exp[1].tolist(),exp[2].tolist(), exp[3].tolist(), exp[4].tolist())

print "\n\n====\n Now testing\n====\n\n"
for episode in xrange(args.num_episodes):
    agent.set_evaluate_flag(True)
    done = False
    world.reset()
    agent.new_episode()
    frame, max_q, min_q, sum_q = 0, 0 - np.Infinity, np.Infinity, 0.0
    actions, act_Vals = [0,0,0,0], [0,0,0,0] #HARD CODED
    while world.is_running() and world.get_time() < args.max_steps_per_episode: 
        frame += 1
        update_cnt += 1

        cur_state = world.get_state()
        action, values = agent.select_action(np.array(cur_state))
        next_state, reward, terminal = world.act(action)
        agent.stash_new_exp(cur_state, action, np.clip(reward,-1,1), terminal, next_state)

        max_q = max(max_q, np.max(values))
        min_q = min(min_q, np.min(values))
        act_Vals += values
        actions[action] += 1
    
    cost = agent.train()

    # REPORTTING
    act_Vals = act_Vals[0]
    if args.verbose >= 0:
        print "%s = ep: %6d:: Re:%5.1f, QMa/Mi/%7.3f/%7.3f,  avg_NSEW:[%7.2f/ %7.2f/ %7.2f/ %7.2f], c: %9.4f, E: %4.3f, W?: %s" % \
            ("{}.{}.{}".format(args.world_id, args.task_id, args.agent_id),
            episode,  world.get_score(), max_q, min_q,
            act_Vals[0]/frame, act_Vals[1]/frame, act_Vals[2]/frame, act_Vals[3]/frame,
            (cost/frame), agent.calculate_epsilon(), "N" if world.is_running() else "Y")
    if args.report_to_sql:
        database.save_episode(
            learner_uuid = learner_uuid, episode = episode, steps_in_episode = frame,
            total_reward = world.get_score(), q_max = max_q, q_min = min_q, 
            avg_action_value_n = act_Vals[0]/frame, avg_action_value_e = act_Vals[1]/frame,
            avg_action_value_s = act_Vals[2]/frame, avg_action_value_w = act_Vals[3]/frame,
            mean_cost = cost/frame,  end_epsilon = agent.calculate_epsilon(),
            did_win = not world.is_running(), 
            is_evaluation=is_eval_episode(episode))
