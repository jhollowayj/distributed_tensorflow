import client_real

def cb(a, b, c):
    print "CALLBACK: {} {} {}".format(a,b,c)
c = client_real.ModDNN_ZMQ_Client()
c.setWeightsAvailableCallback(cb)
c.testServer()
# c.startPolling()