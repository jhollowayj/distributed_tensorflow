import client_real

c = client_real.ModDNN_ZMQ_Client()
def cb(network_type, network_id):
    print "CALLBACK: {} {}".format(network_type, network_id)
    ws = c.requestNetworkWeights(network_type, network_id)
c.setWeightsAvailableCallback(cb)
# c.testServer()
c.startPolling()