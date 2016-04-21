import zmq
import time
import threading
import networks
import numpy as np
import json

class zmq_client():
    def __init__(self):
        # Flags
        self.updatingNetworks = False
        self.usingWeights = False
        self.stop_flag = False
        self.epochs_before_publish_deltas = 10

        
        self.build_client_server_stuff()

    def build_client_server_stuff(self):
        self.context = zmq.Context()
        self.param_server = "localhost"
        self.computerName = "testclient1"
        
        # Broadcast general messages on this socket
        self.message_recv = self.context.socket(zmq.SUB)
        self.message_recv.connect("tcp://%s:5554" % self.param_server)
        self.message_recv.setsockopt(zmq.SUBSCRIBE, b'')

        # this socket receives broadcast parameter updates
        self.param_recv = self.context.socket(zmq.SUB)
        self.param_recv.connect("tcp://%s:5555" % self.param_server)
        self.param_recv.setsockopt(zmq.SUBSCRIBE, b'')

        # this socket can request a parameter update at any time.
        # **Used at startup**
        self.param_rr = self.context.socket(zmq.REQ)
        self.param_rr.connect("tcp://%s:5556" % self.param_server)

        # this socket is where we push our gradient chunks
        self.grad_send = self.context.socket(zmq.PUSH)
        self.grad_send.connect("tcp://%s:5557" % self.param_server)
    
    
    
    
    def test(self):
        self.param_rr.send( json.dumps(['world', 1]) )
        weights = []
        res = self.param_rr.recv()
        ws = res.split("$$SPLIT$$")
        for w in ws:
            weights.append(np.fromstring(w))
        print len(weights)
        
        
        self.param_rr.send( json.dumps(['world', 2]) )
        weights = []
        res = self.param_rr.recv()
        ws = res.split("$$SPLIT$$")
        for w in ws:
            weights.append(np.fromstring(w))
        print len(weights)
        
        
        self.param_rr.send( json.dumps(['agent', 1]) )
        weights = []
        res = self.param_rr.recv()
        ws = res.split("$$SPLIT$$")
        for w in ws:
            weights.append(np.fromstring(w))
        print len(weights)
    
    
    
    
    # ===============================================================
    def publish_deltas(self, deltas_to_publish):
        print "Publishing Deltas is not yet implemented"
        self.grad_send.send("Sending world 1 params from me")
        
        
    def run_network_and_publish_deltas(self):
        delta_count = 0
        while not self.stop_flag:
            ###### Wait for new updates to be set
            if (self.updatingNetworks):
                startTime = time.time()
                while self.updatingNetworks:
                    time.sleep(0.05)
                waittime = time.time() - startTime()
                print "function run_network_and_publish_deltas \
                  waited for {} seconds while the parameters \
                  were being updated from server"\
                  .format(waittime)
             
            ###### Run your networks
            self.usingWeights = True
            # Grab weights
            # Run NN one epoch
            # Calc and Save Deltas
            delta_count += 1
            self.usingWeights = False
            if DeltaCount > self.epochs_before_publish_deltas:
                # Publish deltas
                self.publish_deltas("delta gradients to send")
                delta_count = 0
                # reset Deltas
            

    def recieve_updated_params_from_server(self):
        while not self.stop_flag:
            # Recive params from server into tmp vars
            self.updatingNetworks = True
            while self.usingWeights:
                time.sleep(0.05)
            # Move tmp vars to real network weights
            self.updatingNetworks = False
            
    # ===============================================================
    def stop(self):
        self.stop_flag = True
        
    def start(self):
        # Register with server
        # # Things to do:
        # # * tell server what I want
        # # * get params for how often to send deltas
        # # * register for messages?)
        
        # Start thread to recieve parameter updates back from server
        t = threading.Thread(target=self.recieve_updated_params_from_server)
        t.setDaemon(True)
        t.start()

        # Start thread to run nn, calc gradients, send deltas periodically
        self.run_network_and_publish_deltas()                