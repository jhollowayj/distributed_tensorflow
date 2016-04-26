import zmq
import time
import networks
import numpy as np
import Ops
import json

class zmq_server():
    def __init__(self):
        self.stop_flag = False
        self.build_server_stuff()
        self.build_all_neuralnetworks()
        
    def build_all_neuralnetworks(self):
        self.networks = {}
        self.networks['world'] = {}
        self.networks['task']  = {}
        self.networks['agent'] = {}
        
        self.networks['world'][1] = networks.World1()
        self.networks['world'][2] = networks.World2()
        self.networks['agent'][1] = networks.Agent1()
        
        
    def test(self):
        self.benchmark_network()
    def benchmark_network(self):
        arr = np.array([0.0]*70000000)
        while True:
            print "Server Broadcasting!"
            arr[0] += 1
            start = time.time()
            msg = arr.tostring()
            self.param_bcast.send(msg)
            print "Broadcast over!", arr[0], 1.0 * len(msg) / 1073741824
            time.sleep(0.2)
    def working(self):
        print "starting Server test"
        self.handle_weight_requests()
        self.recieve_delta_weights()
        print "Server going down"

    # ======================================================================
    def build_server_stuff(self):
        self.context = zmq.Context()
        
        

        # Broadcast general messages on this socket
        self.message_bcast = self.context.socket( zmq.PUB )
        self.message_bcast.bind( "tcp://*:5554" )

        # this is the socket we'll use to broadcast our updated parameters
        self.param_bcast = self.context.socket( zmq.PUB )
        self.param_bcast.bind( "tcp://*:5555" )

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

    # ======================================================================
    def handle_weight_requests(self, verbose = True):
        print "Starting to handle initial weight requests"
        # while not self.stop_flag:
        x = 3
        while x > 0:
            x -= 1
            if verbose: print "==Waiting for weight request"
            msg = self.param_rr.recv()
            req = json.loads(msg)
            if verbose: print "  Request header: ", req
            if req[0].encode('ascii') == "request_weights": # Might not need this...
                network_type = req[1].encode('ascii')
                network_type_id = req[2]
                if verbose: print "  Requesting {} {}".format(network_type, network_type_id)

                # Build the network as a string to send all at once.
                weights = self.networks[network_type][network_type_id].get_model_weights()
                if verbose: print "  Building network string of {} layers".format(len(weights))
                compressedWeights = Ops.compress_weights(weights)

                # Send the stuff over!
                if verbose: print "  Sending {} bytes in weights to client".format(len(compressedWeights))
                start = time.time()
                self.param_rr.send(compressedWeights)
                if verbose: print "  Took {} seconds to send those {} bytes.".format(time.time() - start, len(compressedWeights))

            # Elif goes here
            else: #Please don't ever get here.
                print "Aw Snap.  Something went wrong with their request, sending them back an error msg, hope they catch that..."
                self.param_rr.send("ERROR")
        print "Done sending out initial weights"
                
    def recieve_delta_weights(self, verbose = True):
        print "Starting to recieve delta weights"
        x = 3 # Replace x with a while not self.stop_flag 
        while x > 0:
            x -= 1
            if verbose: print "==Waiting on client to send in gradients"
            res = self.grad_recv.recv()
            if verbose: print "  Gradient Recieved", res
        print "Done recieving Gradients"
  
    # ======================================================================
    def start(self):
        # Look into this set up http://zguide.zeromq.org/py:mtserver
    
        print "starting server!"
        while not self.stop_flag:
            socks = dict( poller.poll() )

            if self.param_rr in socks:
                print "  got request for current parameters.  sending..."
                msg = self.param_rr.recv()
                self.param_rr.send( obj.pack_param_msg() )

            if self.grad_recv in socks:
                # receive and accumulate a gradient message
                print "  got grad"
                msg = self.grad_recv.recv()
                obj.unpack_grad_msg( msg )

                # if we've accumulated enough gradient chunks, update our parameters
                if obj.time_to_step():
                    print "doing optimization step"
                    obj.do_opt_step()
                    print "broadcasting new parameters"
                    self.param_bcast.send( obj.pack_param_msg() )
        
        
        

