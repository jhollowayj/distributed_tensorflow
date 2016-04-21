import zmq
import time
import networks
import numpy as np
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
        
        
        
        
        
        
        
        
        
        
        
        
        
    def test(self):
        while True:
            print "starting Server test"
            self.networks['world'][1] = networks.World1()
            self.networks['world'][2] = networks.World2()
            self.networks['agent'][1] = networks.Agent1()
            print "Waiting for request"

            msg = self.param_rr.recv()
            req = json.loads(msg)
            weights = self.networks[req[0].encode('ascii')][req[1]].get_model_weights()
            tosend = "" 
            print len(weights)
            for i, w in enumerate(weights):
                tosend += w.tostring()
                if i < (len(weights) - 1):
                    tosend += "$$SPLIT$$"
            self.param_rr.send(tosend)
                
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
        
        
        

