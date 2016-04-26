class Object(object):
    pass

def getConfig():
    config = Object()
    config.serverHostName = "infinity"
    config.WorldServerHostName = "infinity"
    config.TaskServerHostName = "infinity"
    config.AgentServerHostName = "infinity"        
    config.just_one_server = True
    return config