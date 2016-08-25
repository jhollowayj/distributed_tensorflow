from __future__ import print_function
import os
import socket
from hostlist import expand_hostlist
import tensorflow as tf

## it may be useful to know that slurm_nodeid tells you which node you are one (in case there is more than one task on any given node...)
## Perhaps you could better assign parameter servers be distributed across all nodes before doubleing up on one.
 
def default_port_number():
    if 'SLURM_STEP_RESV_PORTS' in os.environ:
        # Some slurm configureations have the ability to reserve ports, so use that if it's enabled.
        return int(os.environ['SLURM_STEP_RESV_PORTS'].split('-')[0])
    else:
        return 2222

def hostname_list():
    assert 'SLURM_NODELIST' in os.environ
    return expand_hostlist(os.environ['SLURM_NODELIST'])

def my_hostname():
    # I know there is an environ['HOSTNAME'], but it wasn't setup correctly on my slurm cluster.
    # I'm hoping this will be good enough for all users...
    return os.environ['HOSTNAME'] # TODO REMOVE
    return socket.gethostname() # http://stackoverflow.com/questions/4271740
    

def build_cluster_spec(num_param_servers=1, num_workers=None, default_port=default_port_number()):
    hostlist = hostname_list()
    n_comps = len(hostlist)    # len(hostlist) == n_procs() iff the sbatch params --ntasks == --nodes
    if num_workers is None:
        num_workers = n_comps-num_param_servers

    assert num_workers + num_param_servers == n_comps, "ERROR: your number of parameter servers({}) and number of works({}) is different than the number of alloted nodes({})!".format(num_workers, num_param_servers, n_procs)
        
    host_to_role_and_id = {}
    cluster_dictionary = { 'ps': [], 'worker': [] }

    # TODO it might be that someone wants to use a node for more than one process.  
    # If that is the case, we can just round-robin it through the node list.
    #   (Same should go for the workers below.)
    for ps_id in range(num_param_servers):
        host = hostlist[ps_id]
        cluster_dictionary['ps'].append("{}:{}".format(host, default_port))
        host_to_role_and_id[host] = ('ps', ps_id)
        
    for wk_id in range(num_workers):
        host = hostlist[wk_id + num_param_servers]
        cluster_dictionary['worker'].append("{}:{}".format(host, default_port))
        host_to_role_and_id[host] = ("worker", wk_id)
        
    # If we do end up round-robin-ing the computers, we will need a method other than hostname to map ids to the right hostnames.
    # Something like task_
    my_role_and_id = host_to_role_and_id[my_hostname()]
    cluster_spec = tf.train.ClusterSpec(cluster_dictionary)
    return cluster_spec,       my_role_and_id[0], my_role_and_id[1]
    return cluster_dictionary, my_role_and_id[0], my_role_and_id[1]

# def my_proc_id(): # my_node_id should also be available if needed.
#     assert 'SLURM_PROCID' in os.environ
#     return int(os.environ['SLURM_PROCID'])

# def number_slurm_procs():
#     assert 'SLURM_NPROCS' in os.environ
#     return int(os.environ['SLURM_NPROCS'])

   
