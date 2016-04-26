import zmq

from networks import NetworkType, Messages
import zmqconfig
import Ops

class Object(object):
    pass
    
class ModDNN_ZMQ_Client:
    def __init__(self):
        config = zmqconfig.getConfig()
        self.ZMQ_setup(config) 
        
    def ZMQ_setup(self, config):
        self.context = zmq.Context()
        self.poller = zmq.Poller()
        self.servers = {}
        if config.just_one_server:
            server = self.buildServerConnections(config.serverHostName)
            self.servers[NetworkType.World] = server
            self.servers[NetworkType.Task]  = server
            self.servers[NetworkType.Agent] = server
            self.poller.register(server.message_recv, zmq.POLLIN)
        else:
            self.servers[NetworkType.World] = self.buildServerConnections(config.WorldServerHostName)
            self.servers[NetworkType.Task]  = self.buildServerConnections(config.TaskServerHostName)
            self.servers[NetworkType.Agent] = self.buildServerConnections(config.AgentServerHostName)
            for server in self.servers:
                self.poller.register(server.message_recv, zmq.POLLIN)
        
            
    def buildServerConnections(self, param_server):
        server = Object()
        # Broadcast general messages on this socket
        # ** Used to be notified of new weights availbable.**
        server.message_recv = self.context.socket(zmq.SUB)
        server.message_recv.connect("tcp://%s:5555" % param_server)
        server.message_recv.setsockopt(zmq.SUBSCRIBE, b'')

        # this socket can request a parameter update at any time.
        # **Used at startup and throughout execution**
        server.param_rr = self.context.socket(zmq.REQ)
        server.param_rr.connect("tcp://%s:5556" % param_server)

        # this socket is where we push our gradient chunks
        server.grad_send = self.context.socket(zmq.PUSH)
        server.grad_send.connect("tcp://%s:5557" % param_server)
        return server
            
    def sendGradients(self, gradients, network_type = NetworkType.World, network_type_id = 1):
        ''' Send gradients to parameter server defiend in the config function.
            If gradients comes in as a list of networks, we will collapse it for you.
            network_type must be of type NetworkType enum defined in networks.py
            network_type_id isn't checked, but must be non-negative
        '''
        # Error check
        # if not isinstance(network_type, NetworkType):
        #     raise TypeError("network_type must be set to enum value of NetworkType")
        if (network_type_id < 0):
            raise ValueError("network_type_id must be non-negative")
            
        # Compress the weights
        compressedWeights = None
        if isinstance(gradients, list): # TODO is there a better check we can run besides list?
            compressedWeights = Ops.compress_weights(gradients) # Custom Picling
        else:
            compressedWeights = gradients

        # Get the specific server
        server = self.servers[network_type]
            
        # Send the compressed weights
        msg = Ops.label_compressed_weights(
                network_type, network_type_id, compressedWeights)
        server.grad_send.send(msg)
        
        
    def requestNetworkWeights(self, network_type = NetworkType.World, network_type_id = 1):
        ''' Request network Weights
            network_type must be of type NetworkType enum defined in networks.py
            network_type_id isn't checked, but must be non-negative
        '''
        # Error Check
        # if not isinstance(network_type, NetworkType):
        #     raise TypeError("network_type must be set to enum value of NetworkType")
        if (network_type_id < 0):
            raise ValueError("network_type_id must be non-negative")
            
        # Get the specific server
        server = self.servers[network_type]
            
        # Send request
        msg = Ops.compress_request(
                Messages.RequestingNetworkWeights, network_type, network_type_id)
        server.param_rr.send(msg)

        # Receive Response
        response_string = server.param_rr.recv()
        return Ops.decompress_weights(response_string)

    def setWeightsAvailableCallback(self, cb):
        ''' Set function to call for when ZMQ gets a message of available weights
            Simulators should use this to know they can request updated weights (i.e. requestNetworkWeights)
            
            Callback api should be (msg, network, id).
            i.e. (Messages.NetworkWeightsReady, NetworkType.World, 1)
        '''
        self.callback_weights_available = cb
        
    def testServer(self):
        arrs = []
        ### Request Weights
        for i in range(10):
            print "requesting network weights, iteration {}".format(i + 1)
            arr = self.requestNetworkWeights()
            arrs.append(arr)  
        ### Send in gradients!
        for a in arrs:
            self.sendGradients(a)
        
    def startPolling(self):
        self.stop = False
        while not self.stop:
            # check for new network weights
            socks = dict( self.poller.poll( 1000 ) ) # poll param == sleep time between polls
            print "CLIENT_REAL POLLER:", socks
            
            # TODO write the polling code
            self.servers[NetworkType.World]
            