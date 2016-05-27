import time
import client, server
import JacobsMazeWorld, dqn_with_gym
import thread
import numpy as np
from networks import NetworkType, Messages


def server_handle_weight_request(server, times = 1):
    for _ in range(times):
        server.handle_weight_request(server.param_rr)
def server_handle_incoming_gradients(server, times = 1):
    for _ in range(times):
        server.handle_incoming_gradients(server.grad_recv)

def build_needed_objects():
    world = JacobsMazeWorld.JacobsMazeWorld(task_id = 1, world_id = 1)
    agent = dqn_with_gym.Agent(
        state_size=world.get_state_space(),
        number_of_actions=len(world.get_action_space()),
        memory=100000, epsilon=1.0/25.0,
        batch_size=5, anealing_size=600)
    c = client.ModDNN_ZMQ_Client()
    s = server.ModDNN_ZMQ_Server()
    def cb(network_type, network_id):
        agent.set_weights(
            c.requestNetworkWeights(network_type, network_id),
            network_type)
    c.setWeightsAvailableCallback(cb)
    return world, agent, c, s

def test_server_client_sending():
    # Send gradients and make sure the server stores them correctly
    world, agent, c, s = build_needed_objects()
    print "+======================+"
    weights = s.nnetworks[1][1].get_model_weights()
    start = weights[0][0][0]
    s.zero_weights()
    assert s.nnetworks[1][1].get_model_weights()[0][0][0] == 0
    c.sendGradients(weights, NetworkType.World, 1)
    thread.start_new_thread( server_handle_incoming_gradients, (s,) )
    thread.start_new_thread( server_handle_weight_request, (s,) )
    time.sleep(1)
    results = c.requestNetworkWeights(NetworkType.World, 1)
    assert results[0][0][0] == start

def test_server_client_sending_2x():
    # Send gradients and make sure the server stores them correctly
    world, agent, c, s = build_needed_objects()
    print "+======================+"
    weights = s.nnetworks[1][1].get_model_weights()
    start = weights
    # print start
    s.zero_weights()
    assert s.nnetworks[1][1].get_model_weights()[0][0][0] == 0
    c.sendGradients(weights, NetworkType.World, 1)
    c.sendGradients(weights, NetworkType.World, 1)
    thread.start_new_thread( server_handle_incoming_gradients, (s,2) )
    thread.start_new_thread( server_handle_weight_request, (s,) )
    time.sleep(1)
    results = c.requestNetworkWeights(NetworkType.World, 1)
    # print results[0][0][0]
    # print np.array([[1,2],[3,4]]) * 2
    assert  np.array_equal(results[0], start[0] * 2)
    assert  np.array_equal(results[1], start[1] * 2)

test_server_client_sending()
test_server_client_sending_2x()
