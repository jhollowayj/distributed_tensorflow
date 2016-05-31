import time
import numpy as np
import client
import GenericAgent
import JacobsMazeWorld
from networks import NetworkType, Messages


### COMMAND LINE ARGUMENTS ###
import argparse
parser = argparse.ArgumentParser()
# WORLD
parser.add_argument('--random_starting_location', '-rand_start', default=True, action='store_true')
parser.add_argument('--onehot_state', default=False, action='store_true')
# NEURAL-NET
parser.add_argument('--scale_input', default=False, action='store_true')
parser.add_argument('--discount_rate', '-disc', default=0.80, type=float)
parser.add_argument('--learning_rate', '-lr', default=0.0001, type=float)
parser.add_argument('--momentum', '-nnm', default=0.0, type=float) # 0 works well.
parser.add_argument('--requested_gpu_vram_percent', '-vram', default=0.2,  type=float)
parser.add_argument('--device_to_use', '-device', default=1,  type=int)
parser.add_argument('--verbose', '-v', default=0,  type=int)
args = parser.parse_args()
### COMMAND LINE ARGUMENTS ###

### INITIALIZE OBJECTS ###
world = JacobsMazeWorld.JacobsMazeWorld(2,1,1,random_start=True, onehot_state = args.onehot_state)

# x = 6
# y = 6
# print world.act(0, x, y)
# print world.act(1, x, y)
# print world.act(2, x, y)
# print world.act(3, x, y)
# exit()
agent = GenericAgent.Agent(
    state_size=world.get_state_space(),
    number_of_actions=len(world.get_action_space()),
    input_scaling_vector=None, # Default None
    batch_size=250,
    learning_rate = args.learning_rate,
    momentum = args.momentum,
    discount = args.discount_rate,
    boltzman_softmax=False,
    use_experience_replay=False,
    allow_local_nn_weight_updates = True,
    requested_gpu_vram_percent = args.requested_gpu_vram_percent,
    device_to_use = args.device_to_use,
    )

print "\n\nCodename: {}".format("Systematic_Trainer")
### INITIALIZE OBJECTS ###
a = 1
b = 3
games = [
    [2,2,1]
    
    # [1,1,1],
    # [1,1,2],
    # [1,1,3],
    
    # [1,2,1],
    # [1,3,1], # Note: you can't the same layers at the same time.  This setup allows 2 learners @ the same time
    # [1,2,2],
    # [1,3,2], # Note: you can't the same layers at the same time.  This setup allows 2 learners @ the same time
    # [1,2,3],
    # [1,3,3], # Note: you can't the same layers at the same time.  This setup allows 2 learners @ the same time


    # [2,1,1], [3,1,1], # Note: you can't the same layers at the same time.  This setup allows 2 learners @ the same time
    # [2,1,2], [3,1,2], # Note: you can't the same layers at the same time.  This setup allows 2 learners @ the same time
    # [2,1,3], [3,1,3], # Note: you can't the same layers at the same time.  This setup allows 2 learners @ the same time
    # [2,2,1], [3,2,1], # Note: you can't the same layers at the same time.  This setup allows 2 learners @ the same time
    # [2,2,2], [3,2,2], # Note: you can't the same layers at the same time.  This setup allows 2 learners @ the same time
    # [2,2,3], [3,2,3], # Note: you can't the same layers at the same time.  This setup allows 2 learners @ the same time
    # [2,3,1], [3,3,1], # Note: you can't the same layers at the same time.  This setup allows 2 learners @ the same time
    # [2,3,2], [3,3,2], # Note: you can't the same layers at the same time.  This setup allows 2 learners @ the same time
    # [2,3,3], [3,3,3], # Note: you can't the same layers at the same time.  This setup allows 2 learners @ the same time 
]

F = False
T = True
train_layer_flags = [
    [[T, T, T]] * 10
    #  # [T, T, T],
    # [F, F, T], [F, F, T], 
    
    # [F, T, F], [F, T, F],
    # [F, T, F], [F, T, F], 
    # [F, T, F], [F, T, F], 
   
    # [T, F, F], [T, F, F], [T, F, F],
    # [T, F, F], [T, F, F], [T, F, F],
    # [T, F, F], [T, F, F], [T, F, F], 
   
    # [T, F, F], [T, F, F], [T, F, F],
    # [T, F, F], [T, F, F], [T, F, F],
    # [T, F, F], [T, F, F], [T, F, F]
]
times = [5] * len(train_layer_flags)
# times = [
#      5,  5,  5,
#     25, 25, 25,
#     30, 30, 30,
    
#     60, 60, 60,
#     60, 60, 60,
#     60, 60, 60,
    
#     60, 60, 60,
#     60, 60, 60,
#     60, 60, 60
# ]

def Train(ids, train_layer_flags, max_train_time = 60):
    world = JacobsMazeWorld.JacobsMazeWorld( ids[0], ids[1], ids[2],
                                             random_start = args.random_starting_location,
                                             onehot_state = args.onehot_state)
    # tf_client = client.ModDNN_ZMQ_Client(ids[0], ids[1], ids[2])

    # request weights
    # agent.model.set_weights(tf_client.requestNetworkWeights(NetworkType.World), NetworkType.World)
    # agent.model.set_weights(tf_client.requestNetworkWeights(NetworkType.Task),  NetworkType.Task)
    # agent.model.set_weights(tf_client.requestNetworkWeights(NetworkType.Agent), NetworkType.Agent)
    # agent.model.stash_original_weights()
    
    # Build the perfect experience database
    agent.reset_exp_db()
    states = world.get_all_possible_states()
    exp = []
    for state in states:
        for act in world.get_action_space():
            cur_state = world.get_state(state[0], state[1])
            next_state, reward, terminal = world.act(act, state[0], state[1])
            exp.append([cur_state, act, next_state, reward, terminal])
                
    exp = np.array(exp).T
    
    # Now just train for ever!
    prefix = "{}{}.{}{}.{}{}".format(ids[0], "T" if train_layer_flags[0] else "",
                                     ids[1], "T" if train_layer_flags[1] else "",
                                     ids[2], "T" if train_layer_flags[2] else "")
    agent.model.set_train_layer_flags(train_layer_flags)
    agent.train_everything(prefix, max_train_time,
                           exp[0].tolist(), exp[1].tolist(),
                           exp[2].tolist(), exp[3].tolist(),
                           exp[4].tolist(), grads=True, world=world)
    
    #Send gradients back to the server for him to stash
    # grads = agent.model.get_delta_weights()
    # if not train_layer_flags[0]: assert np.sum(grads[0]) == 0 and np.sum(grads[1]) == 0 # I bet this will fail on the second round.
    # if not train_layer_flags[1]: assert np.sum(grads[2]) == 0 and np.sum(grads[3]) == 0
    # if not train_layer_flags[2]: assert np.sum(grads[4]) == 0 and np.sum(grads[5]) == 0
    
    # tf_client.sendGradients(grads[0:2], NetworkType.World)  ##### #   #       #
    # tf_client.sendGradients(grads[2:4], NetworkType.Task)   # These should be fine... if they aren't 0, then the agent.set_layer_locks failed
    # tf_client.sendGradients(grads[4:6], NetworkType.Agent)  ##### #   #       #


completeLoops = 30
for _ in range(completeLoops):
    for i, game in enumerate(games):
        print "\n\n\n\n\n\n\n\n Training: {}, locks:{}".format(games[i], train_layer_flags[i])
        Train(games[i], train_layer_flags[i], times[i])
        time.sleep(3)
