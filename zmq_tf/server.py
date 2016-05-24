import zmq
import time
import numpy as np
import tensorflow as tf
import statistics
import networks as networks, Ops
from networks import NetworkType, Messages

import errno    
import os


class ModDNN_ZMQ_Server:
    def __init__(self, just_one_server=True, serverType=NetworkType.World,
                 server_learning_rate=1, grad_update_cnt_before_send=2, 
                 tensorflow_random_seed=54321, requested_gpu_vram_percent =(1.0/12.0), device_to_use=1,
                 verbose=2, 
                 ckpt_save_interval=50, weights_ckpt_file="/tmp/model.ckpt", load_ckpt_file_on_start=False,
                 server_codename = ""):
        self.config = {
            'just_one_server': just_one_server,
            'grad_update_cnt_before_send': grad_update_cnt_before_send,
            'serverType': serverType,
            'server_learning_rate': server_learning_rate,
            'tensorflow_random_seed': tensorflow_random_seed,
            'requested_gpu_vram_percent': requested_gpu_vram_percent,
            'device_to_use': device_to_use,
            'verbose': verbose,
            'weights_ckpt_file': weights_ckpt_file,
            'ckpt_save_interval': ckpt_save_interval,
            'load_ckpt_file_on_start': load_ckpt_file_on_start,
        }
        self.make_dirs_for_save_file(weights_ckpt_file)
        np.random.seed(self.config['tensorflow_random_seed'])
        self.weight_update_cnt = 0.0
        self.client_announce_cnt = 0.0
        self.buildNetworks()
        print "Servver setup:: ##################### ", self.nnetworks[NetworkType.World][1].get_model_weights()[0].shape
        self.ZMQ_setup()
        self.server_uuid = statistics.get_new_uuid()
        print "======== SERVER using SERVER-UUID:  {} ========\n============================================".format(self.server_uuid)


###############################################################################

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
        device2use = {-1: "/cpu:0", 0: "/gpu:0", 1: "/gpu:1"}[self.config['device_to_use']]
        with tf.device(device2use):
            if self.config['device_to_use'] == -1:
                self.sess = tf.Session()
            else:
                gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=self.config['requested_gpu_vram_percent'],
                                            allow_growth=True, deferred_deletion_bytes=1)
                self.sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

        self.savable_variables = []
        
        self.nnetworks = {}
        self.gradientCnts = {}
        self.nnetworks[NetworkType.World], self.nnetworks[NetworkType.Task], self.nnetworks[NetworkType.Agent] = None, None, None
        self.gradientCnts[NetworkType.World], self.gradientCnts[NetworkType.Task], self.gradientCnts[NetworkType.Agent] = None, None, None
        
        if self.config['just_one_server']:
            self.build_NN_worlds(), self.build_NN_tasks(), self.build_NN_agents()
        else:
            if   self.config['serverType'] == NetworkType.World: self.build_NN_worlds()
            elif self.config['serverType'] == NetworkType.Task:  self.build_NN_tasks()
            elif self.config['serverType'] == NetworkType.Agent: self.build_NN_agents()
            else: raise Error("Ran out of types on the server")

        self.sess.run(tf.initialize_all_variables())
        self.saver = tf.train.Saver()
        tf.get_default_graph().finalize() # TODO see if we can still save/load weights with this here...
        
        print "\nblah\n{}\n".format(self.config['load_ckpt_file_on_start'])
        if self.config['load_ckpt_file_on_start']:
            self.load_weights()
    
        if self.config['verbose'] >= 1:
            print "Initializing _all _variables on server"

###############################################################################
    def make_dirs_for_save_file(self, filename):
        if not os.path.exists(os.path.dirname(filename)):
            try:
                os.makedirs(os.path.dirname(filename))
            except OSError as exc: # Guard against race condition
                if exc.errno != errno.EEXIST:
                    raise

    # TODO make sure this doesn't break when we switch to use separate servers... if we ever get that far.
    def save_weights(self):
        save_path = self.saver.save(self.sess, self.config['weights_ckpt_file'])
        # save_path = self.saver.save(self.savable_variables, self.config['weights_ckpt_file'])
        if self.config['verbose'] >= 1:
            print("Model saved in file: %s" % save_path)

    def load_weights(self):
        print("attempting to load the weights")
        self.saver.restore(self.sess, self.config['weights_ckpt_file'])
        # self.saver.restore(self.savable_variables, self.config['weights_ckpt_file'])
        print("Model restored.")

###############################################################################
###############################################################################
###############################################################################

    def build_NN_worlds(self):
        if self.config['verbose'] >= 1: print "building worlds"
        self.nnetworks[NetworkType.World] = {}
        self.nnetworks[NetworkType.World][1] = networks.World1(self.sess)
        # self.nnetworks[NetworkType.World][2] = networks.World2(self.sess)
        # self.nnetworks[NetworkType.World][3] = networks.World3(self.sess)

        self.savable_variables += self.nnetworks[NetworkType.World][1].savable_vars()
        # self.savable_variables += self.nnetworks[NetworkType.World][2].savable_vars()
        # self.savable_variables += self.nnetworks[NetworkType.World][3].savable_vars()

        self.gradientCnts[NetworkType.World] = {}
        self.gradientCnts[NetworkType.World][1] = self.config['grad_update_cnt_before_send']
        # self.gradientCnts[NetworkType.World][2] = self.config['grad_update_cnt_before_send']
        # self.gradientCnts[NetworkType.World][3] = self.config['grad_update_cnt_before_send']
        
    def build_NN_tasks(self):
        if self.config['verbose'] >= 1: print "building tasks"
        self.nnetworks[NetworkType.Task] = {}
        self.nnetworks[NetworkType.Task][1] = networks.Task1(self.sess)
        self.nnetworks[NetworkType.Task][2] = networks.Task2(self.sess)
        self.nnetworks[NetworkType.Task][3] = networks.Task3(self.sess)

        self.savable_variables += self.nnetworks[NetworkType.Task][1].savable_vars()
        self.savable_variables += self.nnetworks[NetworkType.Task][2].savable_vars()
        self.savable_variables += self.nnetworks[NetworkType.Task][3].savable_vars()

        self.gradientCnts[NetworkType.Task] = {}
        self.gradientCnts[NetworkType.Task][1] = self.config['grad_update_cnt_before_send']
        self.gradientCnts[NetworkType.Task][2] = self.config['grad_update_cnt_before_send']
        self.gradientCnts[NetworkType.Task][3] = self.config['grad_update_cnt_before_send']
    
    def build_NN_agents(self):
        if self.config['verbose'] >= 1: print "building agents"
        self.nnetworks[NetworkType.Agent] = {}
        self.nnetworks[NetworkType.Agent][1] = networks.Agent1(self.sess)
        self.nnetworks[NetworkType.Agent][2] = networks.Agent2(self.sess)
        self.nnetworks[NetworkType.Agent][3] = networks.Agent3(self.sess)

        self.savable_variables += self.nnetworks[NetworkType.Agent][1].savable_vars()
        self.savable_variables += self.nnetworks[NetworkType.Agent][2].savable_vars()
        self.savable_variables += self.nnetworks[NetworkType.Agent][3].savable_vars()

        self.gradientCnts[NetworkType.Agent] = {}
        self.gradientCnts[NetworkType.Agent][1] = self.config['grad_update_cnt_before_send']
        self.gradientCnts[NetworkType.Agent][2] = self.config['grad_update_cnt_before_send']
        self.gradientCnts[NetworkType.Agent][3] = self.config['grad_update_cnt_before_send']

###############################################################################
###############################################################################
###############################################################################
###############################################################################
###############################################################################

    def handle_client_request(self, receiving_socket):
        # Receive the request
        msg = receiving_socket.recv()
        msg_type, network_type, network_id = Ops.decompress_request(msg)
        
        if msg_type == Messages.RequestingNetworkWeights:
            self.send_weights_to_client(network_type, network_id, receiving_socket)
        elif msg_type == Messages.RequestingServerUUID:
            self.send_server_uuid(receiving_socket)
        else:
            print "Error, recieved bad request, throwing it away  ({} == {} equates to {})".format(msg_type, Messages.RequestingNetworkWeights, msg_type == Messages.RequestingNetworkWeights)
            return 
    def send_server_uuid(self, receiving_socket):
        receiving_socket.send(self.server_uuid.bytes)
        
    def send_weights_to_client(self, network_type, network_id, receiving_socket):
        if self.config['verbose'] >= 2:
            print "+++ responing to client request for weights {} {}".format(network_type, network_id)

        # TODO check { netowkr_type, network id against network type }
        # TODO TODO TODO IF there are any errors, we must notify the client and allow them to fail gracefully.  (Ideally, this shouldn't be a problem as we are all among friends here, but you know how it goes...)
        
        # Send the requested weights
        weights = self.nnetworks[network_type][network_id].get_model_weights()
        
        ## Debugging ## if network_type == 1 and network_id == 1: print "============ 1,1[0][0][0]: ", weights[0][0][0] 
        receiving_socket.send( Ops.compress_weights( weights ) )
    
    
    def handle_incoming_gradients(self, receiving_socket):
        # Update the weight_update_cnt
        self.weight_update_cnt += 1.0
        if self.config['verbose'] >= 2: print "### recieving incoming gradients!  (# {})".format(int(self.weight_update_cnt / 3.0)),

        # Receive the request
        network_type, network_id, compressed_gradients = \
            Ops.delabel_compressed_weights(
                receiving_socket.recv())
        gradients = Ops.decompress_weights(compressed_gradients)

        plus, start = np.sum(gradients[0]), np.sum(self.nnetworks[network_type][network_id].get_model_weights()[0])
        if self.config['verbose'] >= 3:
            print "||| Recieved Gradient ({}.{}) {:7.4} * {} = {:7.4}.  x+ {:7.4} => ".format(
                network_type, network_id, plus, self.config['server_learning_rate'], plus * self.config['server_learning_rate'], start),

        for g in gradients:
            g *= self.config['server_learning_rate']

        self.nnetworks[network_type][network_id].add_gradients(gradients)
        if self.config['verbose'] >= 3:
            print "{:7.4}".format( np.sum(self.nnetworks[network_type][network_id].get_model_weights()[0])),
        if self.config['verbose'] >= 2: print

        # Save new weights if needed...
        if self.weight_update_cnt % self.config['ckpt_save_interval'] == 0:
            self.save_weights()
        
        # Update count, publish if needed.
        self.gradientCnts[network_type][network_id] -= 1
        if self.gradientCnts[network_type][network_id] == 0:
            if self.config['verbose'] >= 2:
                print "___ Recieved {} gradient updates for {},{}.  Now pushing new networks!".format( self.config['grad_update_cnt_before_send'], network_type, network_id)
            self.gradientCnts[network_type][network_id] = self.config['grad_update_cnt_before_send']
            self.publish_weights_available(network_type, network_id)
        
        # TODO Maybe send a response saying it went through, or it failed???
    

    def publish_weights_available(self, network_type, network_id):
        # Error check
        # if not isinstance(network_type, NetworkType):
        #     raise TypeError("network_type must be set to enum value of NetworkType")
        self.client_announce_cnt += 1
        if (network_id < 0):
            raise ValueError("network_id must be non-negative")
        if self.config['verbose'] >= 2:
            print "=== ANNOUNCING NETWORK WEIGHT {},{} (# {}): {}".format(network_type, network_id, int(self.client_announce_cnt), self.nnetworks[network_type][network_id].get_model_weights()[0][0][0])

        # Publish a message saying the weights are available! 
        self.message_bcast.send(Ops.compress_msg(Messages.NetworkWeightsReady, network_type, network_id))
    
    def poll_once(self):
        # check for new request or gradients
        socks = dict( self.poller.poll( 500 ) )
        # print "server Polling results: {}".format(socks)
        if self.param_rr in socks:
            self.handle_client_request(self.param_rr)
        if self.grad_recv in socks:
            self.handle_incoming_gradients(self.grad_recv)
        
        # Maybe save the weights to disk periodically?
    def startPolling(self):
        self.stop = False
        while not self.stop:
            self.poll_once()
            
    def zero_weights(self):
        '''Used only for test cases, please don't break stuff with this...'''
        nets = [self.nnetworks[NetworkType.World][1],
                self.nnetworks[NetworkType.Task][1],
                self.nnetworks[NetworkType.Agent][1]]
        for n in nets:
            n.set_model_weights([np.zeros(layer.shape) for layer in n.get_model_weights()])

    def testClient(self):
        self.stop = False
        while not self.stop:
            # check for new request or gradients
            socks = dict( self.poller.poll( 500 ) )
            if self.param_rr in socks:
                print "handling network weight request"
                self.handle_client_request(self.param_rr)
            if self.grad_recv in socks:
                self.handle_incoming_gradients(self.grad_recv)

            # Send out fake messages
            a = np.random.randint(1,4)
            b = np.random.randint(1,3)
            print "pushing {}, {}".format(a,b)
            self.publish_weights_available(a, b)