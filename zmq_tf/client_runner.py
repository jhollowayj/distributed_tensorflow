import numpy as np
import time
import os, sys
sys.path.append(os.path.abspath(os.path.join('..', 'simpleMazeWorlds')))

import client
import dqn_with_gym
import JacobsMazeWorld
from networks import NetworkType, Messages

### COMMAND LINE ARGUMENTS ###
import argparse
parser = argparse.ArgumentParser()
# SHARED
parser.add_argument('--world_id', '-wid', default=1, type=int, required=False, help="ID of the world(maze) you want to use")
parser.add_argument('--task_id', '-tid',  default=1, type=int, required=False, help="ID of the task(start/end positions) you want to use")
parser.add_argument('--agent_id', '-aid', default=1, type=int, required=False, help="ID of the agent you want to use (nsew/sewn/ewns/etc")
# AGENT
parser.add_argument('--num_episodes', '-ne', default=10000,  type=int, required=False, help="")
parser.add_argument('--annealing_size', '-an', default=800,  type=int, required=False, help="")
parser.add_argument('--exp_replay_memory', '-mem', default=10000,  type=int, required=False, help="")
parser.add_argument('--epsilon', '-e', default=0.04,  type=float, required=False, help="")
parser.add_argument('--observer', '-o', default=False,  type=bool, required=False, help="")
# NEURAL-NET            # Discount Factor, Learning Rate, etc. TODO
parser.add_argument('--allow_local_nn_weight_updates', '-nnu', default=False,  type=bool, required=False, help="")
parser.add_argument('--requested_gpu_vram_percent', '-vram', default=0.01,  type=float, required=False, help="")
# RUNNER
parser.add_argument('--max_steps_per_episode', '-msteps', default=200,  type=int, required=False, help="")
parser.add_argument('--write_csv', default=False,  type=bool, required=False, help="")
parser.add_argument('--csv_filename', default="tmp_res.csv",  type=bool, required=False, help="")
# CLIENT-SERVER
parser.add_argument('--gradients_until_send', '-grads', default=500,  type=int, required=False, help="")
parser.add_argument('--ignore_server', default=False,  type=bool, required=False, help="")
args = parser.parse_args()
### COMMAND LINE ARGUMENTS ###

### OTHER FUNCTIONS ###
def send_gradients():
    # print "     sending Gradients!!!!"
    s = time.time()
    grads = agent.get_gradients()
    tf_client.sendGradients(grads[0], NetworkType.World)
    tf_client.sendGradients(grads[1], NetworkType.Task)
    tf_client.sendGradients(grads[2], NetworkType.Agent)
    end = time.time()
    print "   Time to send gradients was {} seconds".format(end-s)
def cb(network_type, network_id):
    # print "CALLBACK: {} {}".format(network_type, network_id)
    ws = tf_client.requestNetworkWeights(network_type)
    agent.set_weights(ws, network_type)
### OTHER FUNCTIONS ###

### INITIALIZE OBJECTS ###
world = JacobsMazeWorld.JacobsMazeWorld(
    world_id = args.world_id,
    task_id = args.task_id,
    agent_id = args.agent_id)
    
tf_client = client.ModDNN_ZMQ_Client(
    world_id = args.world_id,
    task_id = args.task_id,
    agent_id = args.agent_id)
    
    
agent = dqn_with_gym.Agent(
    state_size=world.get_state_space(),
    number_of_actions=len(world.get_action_space()),
    input_scaling_vector=world.get_state__maxes(),
    memory=args.exp_replay_memory,
    epsilon=args.epsilon,
    batch_size=5,
    annealing_size=int(args.annealing_size), # annealing_size=args.annealing_size,
    allow_local_nn_weight_updates = args.allow_local_nn_weight_updates,
    requested_gpu_vram_percent = args.requested_gpu_vram_percent,
    )

if not args.ignore_server:
    tf_client.setWeightsAvailableCallback(cb)
    #Request and set initial weights
    agent.set_weights(tf_client.requestNetworkWeights(NetworkType.World), NetworkType.World)
    agent.set_weights(tf_client.requestNetworkWeights(NetworkType.Task),  NetworkType.Task)
    agent.set_weights(tf_client.requestNetworkWeights(NetworkType.Agent), NetworkType.Agent)
### INITIALIZE OBJECTS ###

### RUN !!!!!!!!!!!!! ###
if args.write_csv:
    csv = open(args.csv_filename,'w', 0)
    csv.write("episode,total_reward,mean_cost,max_q,endEpsilon,didFinish\n")
starttime = time.time()
update_cnt = 1
for episode in xrange(args.num_episodes):
    done = False
    world.reset()
    agent.new_episode()
    total_cost, frame, max_q = 0.0, 0, 0 - np.Infinity 
    arr, actions = world.heatmap_adder(), [0,0,0,0]
    while world.is_running() and world.get_time() < args.max_steps_per_episode: 
        frame += 1
        update_cnt += 1

        cur_state = world.get_state()
        action, values = agent.select_action(np.array(cur_state))
        reward = world.act(action)
        total_cost += agent.train(reward) # IS this where it goes wrong?

        max_q = max(max_q, np.max(values))
        actions[action] += 1
        arr += world.heatmap_adder()
    
        if not args.ignore_server and update_cnt % args.gradients_until_send == 0:
            send_gradients()
        if not args.ignore_server:
            tf_client.poll_once() # calls the callback added above if weights available!
    # REPORTTING
    runtime = time.time() - starttime
    totaltime = runtime / (episode+1) * args.num_episodes

    print "episode: %6d::%4d/%4ds:: total reward: %6.3f, mean cost: %13.9f, max_q: %10.6f, endEpsilon: %4.3f, didFinish: %s" % \
          (episode, runtime, totaltime, world.get_score(), (total_cost/frame),
           max_q, agent.calculate_epsilon(), "No" if world.is_running() else "Yes") 
    if args.write_csv:
        csv.write("{},{},{},{},{},{}\n".format(e, world.get_score(), (total_cost/frame), max_q, agent.calculate_epsilon(), 0 if world.is_running() else 1))

if args.write_csv: csv.close()
