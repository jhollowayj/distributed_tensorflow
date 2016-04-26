import zmq

from networks import NetworkType, Messages
import networks
import zmqconfig
import Ops
import numpy as np
import time

# TODO add verbose flags for logging everything
class Object(object):
    pass
    
class ModDNN_ZMQ_Server:
    def __init__(self):
        
        config = Object()
        config.just_one_server = True
        config.grad_update_cnt = 25
        config.serverType = NetworkType.World
        
        print NetworkType.World,NetworkType.Task,NetworkType.Agent
        self.config = config
        self.buildNetworks()
        self.ZMQ_setup()
       
    def ZMQ_setup(self):
        self.context = zmq.Context()
        self.poller = zmq.Poller()
        
        # Broadcast general messages on this socket
        self.message_bcast = self.context.socket( zmq.PUB )
        self.message_bcast.bind( "tcp://*:5555" )

        # this is a point-to-point socket we'll use to send parameter
        # updates when a new client starts up
        self.param_rr = self.context.socket( zmq.REP )
        self.param_rr.bind( "tcp://*:5556" )

        # this is the socket we use to receive gradient updates from clients
        self.grad_recv = self.context.socket( zmq.PULL )
        self.grad_recv.bind( "tcp://*:5557" )

        self.poller = zmq.Poller()
        self.poller.register( self.param_rr, zmq.POLLIN )
        self.poller.register( self.grad_recv, zmq.POLLIN )

    def buildNetworks(self):
        self.nnetworks = {}
        self.gradientCnts = {}
        self.nnetworks[NetworkType.World] = None
        self.nnetworks[NetworkType.Task]  = None
        self.nnetworks[NetworkType.Agent] = None
        self.gradientCnts[NetworkType.World] = None
        self.gradientCnts[NetworkType.Task]  = None
        self.gradientCnts[NetworkType.Agent] = None
        
        if self.config.just_one_server:
            self.build_NN_worlds()
            self.build_NN_tasks()
            self.build_NN_agents()
        else:
            if self.config.serverType == NetworkType.World:
                self.build_NN_worlds()
            elif self.config.serverType == NetworkType.Task:
                self.build_NN_tasks()
            elif self.config.serverType == NetworkType.Agent:
                self.build_NN_agents()
            else:
                raise Error("Ran out of types on the server")

    def build_NN_worlds(self):
        print "building worlds"
        self.nnetworks[NetworkType.World] = {}
        self.nnetworks[NetworkType.World][1] = networks.World1()
        self.nnetworks[NetworkType.World][2] = networks.World2()
        self.nnetworks[NetworkType.World][3] = networks.World3()

        self.gradientCnts[NetworkType.World] = {}
        self.gradientCnts[NetworkType.World][1] = self.config.grad_update_cnt
        self.gradientCnts[NetworkType.World][2] = self.config.grad_update_cnt
        self.gradientCnts[NetworkType.World][3] = self.config.grad_update_cnt
        
    def build_NN_tasks(self):
        print "building tasks"
        self.nnetworks[NetworkType.Task] = {}
        self.nnetworks[NetworkType.Task][1] = networks.Task1()
        self.nnetworks[NetworkType.Task][2] = networks.Task2()
        self.nnetworks[NetworkType.Task][3] = networks.Task3()

        self.gradientCnts[NetworkType.Task] = {}
        self.gradientCnts[NetworkType.Task][1] = self.config.grad_update_cnt
        self.gradientCnts[NetworkType.Task][2] = self.config.grad_update_cnt
        self.gradientCnts[NetworkType.Task][3] = self.config.grad_update_cnt
    
    def build_NN_agents(self):
        print "building agents"
        self.nnetworks[NetworkType.Agent] = {}
        self.nnetworks[NetworkType.Agent][1] = networks.Agent1()
        self.nnetworks[NetworkType.Agent][2] = networks.Agent2()
        self.nnetworks[NetworkType.Agent][3] = networks.Agent3()

        self.gradientCnts[NetworkType.Agent] = {}
        self.gradientCnts[NetworkType.Agent][1] = self.config.grad_update_cnt
        self.gradientCnts[NetworkType.Agent][2] = self.config.grad_update_cnt
        self.gradientCnts[NetworkType.Agent][3] = self.config.grad_update_cnt
        
    
    def handle_weight_request(self, receiving_socket):
        # Receive the request
        msg = receiving_socket.recv()
        msg_type, network_type, network_id = Ops.decompress_request(msg)
        print " responing to weight request {} {}".format(network_type, network_id)
        
        # Error check
        if not msg_type == Messages.RequestingNetworkWeights:
            print "Error, recieved bad request, throwing it away"
        # TODO check netowkr_type
        # TODO check network id against network type
        # TODO TODO TODO IF there are any errors, we must notify the
        # # client and allow them to fail gracefully.  (Ideally, 
        # # this shouldn't be a problem as we are all among friends
        # # here, but you know how it goes...)
        
        # Send the requested weights
        weights = self.nnetworks[network_type][network_id].get_model_weights()
        # for w in weights:
        #     print w.shape
        # print weights
        receiving_socket.send(
            Ops.compress_weights( weights ) )
    
    def handle_incoming_gradients(self, receiving_socket):
        print " recieving incoming gradients!" 
        # Receive the request
        network_type, network_id, compressed_gradients = \
            Ops.delabel_compressed_weights(
                receiving_socket.recv())

        gradients = Ops.decompress_weights(compressed_gradients)
        cur_weights = self.nnetworks[network_type][network_id].get_model_weights()
        
        # Error Check
        if len(gradients) != len(cur_weights):
            print "Error, received bad gradients, perhaps they are misslabeled!  (throwing them away)"
            return
        for i in range(len(gradients)):
            if len(gradients[i]) != len(cur_weights[i]):
                print "Error, received bad gradients, perhaps they are misslabeled!  (throwing them away)"
                return
            
        # Build updated weights
        new_weights = []
        for i in range(len(gradients)):
            new_weights.append(gradients[i] + cur_weights[i])
        
        # Apply these weights to the network now!
        self.nnetworks[network_type][network_id].set_model_weights(new_weights)
        
        # Update count, publish if needed.
        self.gradientCnts[network_type][network_id] -= 1
        if self.gradientCnts[network_type][network_id] == 0:
            self.gradientCnts[network_type][network_id] = self.config.grad_update_cnt
            self.publish_weights_available(network_type, network_id)
            
        # TODO Maybe send a response saying it went through, or it failed???
    
    
    def publish_weights_available(self, network_type, network_id):
        # Error check
        # if not isinstance(network_type, NetworkType):
        #     raise TypeError("network_type must be set to enum value of NetworkType")
        if (network_id < 0):
            raise ValueError("network_id must be non-negative")

        # Publish a message saying the weights are available! 
        self.message_bcast.send(Ops.compress_msg(Messages.NetworkWeightsReady, network_type, network_id))
    
    def testClient(self):
        self.stop = False
        while not self.stop:
            # check for new request or gradients
            socks = dict( self.poller.poll( 500 ) )
            if self.param_rr in socks:
                print "handling network weight request"
                self.handle_weight_request(self.param_rr)
            if self.grad_recv in socks:
                self.handle_incoming_gradients(self.grad_recv)

            # Send out fake messages
            a = np.random.randint(1,4)
            b = np.random.randint(1,3)
            print "pushing {}, {}".format(a,b)
            self.publish_weights_available(a, b)


    def startPolling(self):
        self.stop = False
        while not self.stop:
            # check for new request or gradients
            socks = dict( self.poller.poll( 0 ) )
            if self.param_rr in socks:
                self.handle_weight_request(self.param_rr)
            if self.grad_recv in socks:
                self.handle_incoming_gradients(self.grad_recv)
            
            # Maybe save the weights to disk periodically?