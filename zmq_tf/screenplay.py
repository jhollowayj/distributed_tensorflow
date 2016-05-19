import subprocess
import time
import numpy as np
import argparse
import sys

parser = argparse.ArgumentParser()
# SHARED
parser.add_argument('--single_solo', '-solo', default=False, action='store_true', required=False,
                    help="runs a single client without a server, the original test to run (TODO add exp/noexp options)")
parser.add_argument('--single_w_server', '-single', default=False, action='store_true', required=False,
                    help="Branch of the single_solo option, but shares weights to a server who hands them back, 1-to-1 send and recieves.")
parser.add_argument('--agents6games2', '-a6xg2', default=False, action='store_true', required=False,
                    help="Launches 6 agents on 2 games")

if len(sys.argv)==1: # If no arguments, display the help and quit!
    parser.print_help()
    sys.exit(1)
args = parser.parse_args()

###############################################################################
###############################################################################
###############################################################################


pause = False
open_terminal_command = "gnome-terminal -x bash -c 'pwd; {}{}'".format(
    '{}',
    '; echo "\nPress any key to close"; read' if pause else '')
path = "~/research/modularDNN_Practice/zmq_tf/"
# execute_command = 'ls; sleep 5'

###############################################################################
def launch_command(command):
    subprocess.call(open_terminal_command.format(command), shell=True)
    
def launch_client(args):
    time.sleep(0.1)
    launch_command("python {}client_runner.py {}".format(path, args))
    
def launch_server(args):
    time.sleep(1)
    launch_command("python {}server_runner.py {}".format(path, args))
    
###############################################################################

def calc_grad_sends_ratios(N_agents):
    grads = N_agents * 2 # 10 clients, send every 20 grad, keeps server happy.
    sends =  N_agents * 4
    return  grads, sends
def id_args(ids):
    return "-wid {} -tid {} -aid {}".format(ids[0], ids[1], ids[2])
def codename_creator(ids, nTotalAgents, nAgentsOnThisGame):
    return "{}TotalAgents_{}AgentsOn:{}.{}.{}".format(nTotalAgents, nAgentsOnThisGame, ids[0], ids[1], ids[2])

###############################################################################

def test_1_agent_1_world__no_server():
    launch_client('-is -sql -exp -nnu --codename "singleagent.singlegame.noserver"')

def test_1_agent_1_game():
    launch_client('-grads 1 -exp -sql --codename "singleagent.singlegame.withserver"')
    launch_server('-v 3 -send 1')
    
def test_N_agent_1_game(N_agents = 1):
    grads, sends = calc_grad_sends_ratios(N_agents)

    for _ in range(N_agents-1):
        launch_client('-grads {} -npar {}"'.format( grads, N_agents )) # No sql, no name for slaves
    launch_client('-grads {} -npar {} -sql --codename "singleagent.singlegame.withserver"'.format(grads, N_agents))

    launch_server('-v 3 -send {}'.format(sends))

def test_mulitagent_multigame(games_settings=[[1,1,1]], num_players=[1]):
    total_agents = np.sum(num_players)
    for i, game_settings in enumerate(games_settings):
        grads, sends = calc_grad_sends_ratios(num_players[i])
        for slave in range(num_players[i] - 1):
            launch_client('{} -grads {} -npar {}'.format( id_args(game_settings), grads, num_players[i])) # No sql, no name for slaves
        launch_client('{} -grads {} -npar {} -sql --codename "{}"'.format(
            id_args(game_settings), grads, num_players[i],
            codename_creator(game_settings, total_agents, num_players[i])))

    launch_server('-v 3 -send {}'.format(sends))

###############################################################################


if args.single_solo:
    test_1_agent_1_world__no_server()
if args.single_w_server:
    test_1_agent_1_game()
if args.agents6games2:
    test_mulitagent_multigame([[1,1,1],[1,1,2]], [3,3])