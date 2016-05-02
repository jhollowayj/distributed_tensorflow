class Object(object):
    pass

def getConfig():
    config = Object()
    config.serverHostName = "localhost"
    config.WorldServerHostName = "localhost"
    config.TaskServerHostName = "localhost"
    config.AgentServerHostName = "localhost"        
    config.just_one_server = True
    return config