import numpy as np
import time
import os, sys
sys.path.append(os.path.abspath(os.path.join('..', 'simpleMazeWorlds')))

import client
import dqn_with_gym
import JacobsMazeWorld
from networks import NetworkType, Messages


num_episodes = 20000
pleaseRender = False

world = JacobsMazeWorld.JacobsMazeWorld(
    task_id = 1,
    world_id = 1) # use default world params (w1, t2 (bl-tr), a3)
tf_client = client.ModDNN_ZMQ_Client() # use defaults here (w1, t1, a1)
agent = dqn_with_gym.Agent(
    state_size=world.get_state_space(),
    number_of_actions=len(world.get_action_space()),
    memory=100000,
    epsilon=1.0/50.0,
    batch_size=5,
    anealing_size=800,
    just_greedy = False,
    input_scaling_vector=world.get_state__maxes())

def cb(network_type, network_id):
    # print "CALLBACK: {} {}".format(network_type, network_id)
    ws = tf_client.requestNetworkWeights(network_type, network_id)
    agent.set_weights(ws, network_type)
tf_client.setWeightsAvailableCallback(cb)

#     = Request initial weights
tf_client.requestNetworkWeights(NetworkType.World, 1)
tf_client.requestNetworkWeights(NetworkType.Task, 1)
tf_client.requestNetworkWeights(NetworkType.Agent, 1)
# End = Request initial weights


# runId = "20t_ProbE_Rp100n10"
# csv = open('results_{}.csv'.format(runId),'w', 0)
# csv.write("episode,total_reward,mean_cost,max_q,endEpsilon,didFinish\n")
starttime = time.time()
update_cnt = 1
for e in xrange(num_episodes):
    world.reset()
    done = False
    agent.new_episode()
    total_cost = 0.0
    frame = 0
    max_q = 0 - np.Infinity
    arr = world.heatmap_adder()
    actions = [0,0,0,0]
    while world.is_running() and world.get_time() < 200: 
        frame += 1
        update_cnt += 1

        cur_state = world.get_state()
        action, values = agent.select_action(np.array(cur_state))
        reward = world.act(action)
        total_cost += agent.train(reward) # IS this where it goes wrong?

        max_q = max(max_q, np.max(values))
        actions[action] += 1
        arr += world.heatmap_adder()
    
        if update_cnt % 500 == 0: # Now 
            # print "     sending Gradients!!!!"
            s = time.time()
            grads = agent.get_gradients()
            tf_client.sendGradients(grads[0], NetworkType.World, 1)
            tf_client.sendGradients(grads[1], NetworkType.Task, 1)
            tf_client.sendGradients(grads[2], NetworkType.Agent, 1)
            end = time.time()
            print "   Time to send gradients was {} seconds".format(end-s)
    # Client Server stuff:  Check if there are new weights in between games...
    tf_client.poll_once() # calls the callback added above if needed!
    # REPORTTING
    runtime = time.time() - starttime
    totaltime = runtime / (e+1) * num_episodes
    

    print "episode: %6d::%4d/%4ds:: total reward: %6.3f, mean cost: %13.9f, max_q: %10.6f, endEpsilon: %4.3f, didFinish: %s" % \
          (e, runtime, totaltime, world.get_score(), (total_cost/frame),
           max_q, agent.calculate_epsilon(), "No" if world.is_running() else "Yes") 

    # csv.write("{},{},{},{},{},{}\n".format(e, world.get_score(), (total_cost/frame), max_q, agent.calculate_epsilon(), 0 if world.is_running() else 1))

# csv.close()
