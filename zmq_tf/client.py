import zmq

from networks import NetworkType, Messages
import zmqconfig
import Ops

class Object(object):
    pass
    
class ModDNN_ZMQ_Client:
    def __init__(self, world_id = 1, task_id = 1, agent_id = 1):
        config = zmqconfig.getConfig()
        self.ZMQ_setup(config) 
        self.world_id = 1
        self.task_id = 1
        self.agent_id = 1
        self.network_id_lookup = {
            NetworkType.World: self.world_id,
            NetworkType.Task : self.task_id,
            NetworkType.Agent: self.agent_id,
        }
        
    def ZMQ_setup(self, config):
        self.context = zmq.Context()
        self.poller = zmq.Poller()
        self.servers = {}
        if config.just_one_server:
            print "Adding just one server", config.serverHostName
            server = self.buildServerConnections(config.serverHostName)
            self.servers[NetworkType.World] = server
            self.servers[NetworkType.Task]  = server
            self.servers[NetworkType.Agent] = server
            self.poller.register(server.message_recv, zmq.POLLIN)
        else:
            print "Adding several server"
            self.servers[NetworkType.World] = self.buildServerConnections(config.WorldServerHostName)
            self.servers[NetworkType.Task]  = self.buildServerConnections(config.TaskServerHostName)
            self.servers[NetworkType.Agent] = self.buildServerConnections(config.AgentServerHostName)
            for server in self.servers:
                print "(Note: adding multiple server.msg_recv to poller)"
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
            
    def sendGradients(self, gradients, network_type = NetworkType.World):
        ''' Send gradients to parameter server defiend in the config function.
            If gradients comes in as a list of networks, we will collapse it for you.
            network_type must be of type NetworkType enum defined in networks.py
            network_type_id isn't checked, but must be non-negative
        '''
        # Error check
        # if not isinstance(network_type, NetworkType):
        #     raise TypeError("network_type must be set to enum value of NetworkType")
        if (self.network_id_lookup[network_type] < 0):
            raise ValueError("network_type_id must be non-negative, was {}".format(
                self.network_id_lookup[network_type]))
            
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
                network_type, self.network_id_lookup[network_type], compressedWeights)
        server.grad_send.send(msg)
        
    callback_weights_available = None
    
    def requestNetworkWeights(self, network_type = NetworkType.World):
        ''' Request network Weights
            network_type must be of type NetworkType enum defined in networks.py
            network_type_id isn't checked, but must be non-negative
        '''
        print "requesting network weights...",
        # Error Check
        # if not isinstance(network_type, NetworkType):
        #     raise TypeError("network_type must be set to enum value of NetworkType")
        if (self.network_id_lookup[network_type] < 0):
            raise ValueError("network_type_id must be non-negative, was {}".format(
                self.network_id_lookup[network_type]))
            
        # Get the specific server
        server = self.servers[network_type]
            
        # Send request
        msg = Ops.compress_request(
                Messages.RequestingNetworkWeights, network_type, self.network_id_lookup[network_type])
        server.param_rr.send(msg)

        # Receive Response
        response_string = server.param_rr.recv()
        print " network weights recieved!"
        return Ops.decompress_weights(response_string)

    def handle_message(self, socket):
        msg_type, network_type, network_id = Ops.decompress_msg(socket.recv())
        if (network_type == NetworkType.World and network_id == self.world_id) or\
           (network_type == NetworkType.Task and network_id == self.task_id) or\
           (network_type == NetworkType.Agent and network_id == self.world_id):
            if self.callback_weights_available == None:
                print "No callback set, Ignoring the update available"
                return
            else:
                self.callback_weights_available(network_type, network_id)
        
        

    def setWeightsAvailableCallback(self, cb):
        ''' Set function to call for when ZMQ gets a message of available weights
            Simulators should use this to know they can request updated weights (i.e. requestNetworkWeights)
            
            Callback api should be (network, id).
            i.e. (Messages.NetworkWeightsReady, NetworkType.World, 1)
        '''
        self.callback_weights_available = cb
        
    def testServer(self):
        self.startPolling() # I think that's it...
        
    def poll_once(self):
        # check for new network weights
        socks = dict( self.poller.poll( 0 ) ) # poll param == sleep time between polls
        for server in self.servers.values():
            if server.message_recv in socks:
                self.handle_message(server.message_recv)
        
        # I think the training will happen outside this file,
        # So I think this is all we need here.
        
    def startPolling(self):
        self.stop = False
        while not self.stop:
            self.poll_once()
            