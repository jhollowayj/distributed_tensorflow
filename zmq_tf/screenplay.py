import subprocess
import time
import numpy as np
import numpy.random
import random
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
parser.add_argument('--agents2games1', '-a2xg1', default=False, action='store_true', required=False,
                    help="Launches 2 agents on the same game")
parser.add_argument('--agents4games1', '-a4xg1', default=False, action='store_true', required=False,
                    help="Launches 2 agents on the same game")
parser.add_argument('--doResearch', '-all', default=False, action='store_true', required=False,
                    help="Launches multiple instances to train different sections of the networks")
parser.add_argument('--test_single_worlds', '-tsw', default=False, action='store_true', required=False)
parser.add_argument('--comp_softmax', '-csm', default=False, action='store_true', required=False)
parser.add_argument('--just_evaluators','-evals', default=False, action='store_true', required=False)
if len(sys.argv)==1: # If no arguments, display the help and quit!
    parser.print_help()
    sys.exit(1)
args = parser.parse_args()

###############################################################################
###############################################################################
###############################################################################

gpus = [1,0]
gpu_id = 0
extraLongGame = False

pause = False
open_terminal_command = "gnome-terminal -x bash -c 'pwd; {}{}'".format(
    '{}',
    '; echo "\nPress any key to close"; read' if pause else '')
path = "~/research/modularDNN_Practice/zmq_tf/"
# execute_command = 'ls; sleep 5'

###############################################################################
def launch_command(command):
    subprocess.call(open_terminal_command.format(command), shell=True)

def launch_client(args, randEpsilon=True, longGame=True, randDevice=True):
    global gpu_id
    if randEpsilon:
        args = "-e {} ".format(random.uniform(0.01,0.25)) + args
    if longGame:
        if extraLongGame:
            args = "-an 150000 -ns 75000000 " + args # 100x # probably too long...
        else:
            args = "-an 15000 -ns 7500000 " + args # 10x
    if randDevice:
        args = "-device {} ".format(gpus[gpu_id]) + args
        gpu_id = (gpu_id + 1) % len(gpus)
    print "Launching Client W/ Args: {}".format(args)
    # time.sleep(0.2)
    launch_command("python {}client_runner.py {}".format(path, args))
    
def launch_server(args):
    time.sleep(2)
    launch_command("python {}server_runner.py -device 0 {}".format(path, args))
    
###############################################################################

def calc_grad_sends_ratios(N_agents):
    grads = 16 # 16 seems to work well...
    sends = N_agents # space it out too far causes explosions...
    return  grads, N_agents

def id_args(ids):
    return "-wid {} -tid {} -aid {}".format(ids[0], ids[1], ids[2])

def codename_creator(ids, nTotalAgents, nAgentsOnThisGame):
    return "{}TotalAgents_{}AgentsOn:{}.{}.{}".format(nTotalAgents, nAgentsOnThisGame, ids[0], ids[1], ids[2])

###############################################################################

def test_1_agent_1_world__no_server(ids=[1,1,1]):
    launch_client('{} -is -sql -exp -nnu --codename "singleagent.singlegame.noserver.{}.{}.{}"'.format(id_args(ids), ids[0],ids[1],ids[2]), longGame=False)

def test_1_agent_1_game():
    launch_client('-grads 1 -exp -sql --codename "singleagent.singlegame.withserver"', longGame=False) # TODO add in -sql once it's working
    launch_server('-v 2 -send 1')
    
def test_N_agent_1_game(N_agents = 1):
    grads, sends = calc_grad_sends_ratios(N_agents)

    for _ in range(N_agents-1):
        launch_client('-grads {} -npar {} -an 2500'.format( grads, N_agents ), longGame=False) # No sql, no name for slaves
    launch_client('-grads {} -npar {} -sql -an 2500 --codename "{}agents.singlegame.CGrad:{}.ServerUpdate:{}"'.format(grads, N_agents, N_agents, grads, sends), longGame=False)

    launch_server('-v 3 -send {}'.format(sends))

###############################################################################

def test_mulitagent_multigame(games_settings=[[1,1,1]], num_players=[1], total_agents=0, start_server=True):
    for i, game_setting in enumerate(games_settings):
        grads, sends = calc_grad_sends_ratios(num_players[i])
        for slave in range(num_players[i] - 1):
            launch_client('{} -grads {} -npar {}'.format( id_args(game_setting), grads, total_agents)) # No sql, no name for slaves
        launch_client('{} -grads {} -sql -npar {} --codename "{}"'.format( # TODO Add in -sql once it's working
            id_args(game_setting), grads, total_agents,
            codename_creator(game_setting, total_agents, num_players[i])))
            
    if start_server:
        launch_server('-v 3 -send {}'.format(sends))

def client_evaluators(games_settings=[[1,1,1]], total_agents=0):
    for game_setting in games_settings:
        launch_client('{} -npar {}  -o --codename "{}.Evaluator"'.format( # TODO Add in -sql once it's working
            id_args(game_setting), total_agents,
            codename_creator(game_setting, total_agents, -1)))
def client_evaluators_with_training(games_settings=[[1,1,1]], total_agents=0):
    for game_setting in games_settings:
        launch_client('{} -npar {}  -grads 12 --codename "{}.Evaluator"'.format( # TODO Add in -sql once it's working
            id_args(game_setting), total_agents,
            codename_creator(game_setting, total_agents, -1)))
    
###############################################################################


if args.single_solo:
    test_1_agent_1_world__no_server()
if args.single_w_server:
    test_1_agent_1_game()
if args.agents6games2:
    test_mulitagent_multigame([[1,1,1],[1,1,2]], [4,4])
if args.agents2games1:
    test_N_agent_1_game(2)
if args.agents4games1:
    test_N_agent_1_game(8)
if args.doResearch:
    num_trainers_per_game = 3
    train_games = [
        [1,1,1], [1,1,2],
        [1,2,2], [1,2,3],
        [1,3,1], [1,3,3]
    ]
    test_games = [
        [1,1,3],
        [1,2,1],
        [1,3,2]
    ]
    total_agent_count = num_trainers_per_game*len(train_games) + len(test_games)
    # Launch! 
    test_mulitagent_multigame(train_games, [num_trainers_per_game]*len(train_games), total_agent_count, start_server=False)
    client_evaluators(test_games, total_agent_count) # Launch clients to evaluate unseen paths
    launch_server('-v 3 -send {}'.format(num_trainers_per_game*len(train_games))) # Launch Server

if args.just_evaluators:
    test_games = [ [1,1,3], [1,2,1], [1,3,2] ]
    # client_evaluators(test_games, 21)
    client_evaluators_with_training(test_games, 21)
if args.test_single_worlds:

    wid, tid, aid = 1, 3, 1;
    for lr in [0.0002, 0.0001, 0.0008]: # So far 0.0002 is the best
    # for lr in [0.0002, 0.002, 0.02]:
        for m in [0.0, 0.95, 0.99]: # So far 0 is the best
        # for m in [0.0, 0.25, 0.6, 0.9]:
            for scale in [False]: # So far, false is the best
                launch_client(' -is -sql -nnu -wid 1 -tid 3 -aid 1 --codename "paramsweek.1.3.1.LR:{}.M:{}.ScaleInput:{}" -lr {} -nnm {} {}'.format(
                    lr, m, int(scale), lr, m, "--scale_input" if scale else "", longGame=False), randEpsilon=False, longGame=False)

    # test_1_agent_1_world__no_server([1,1,1])
    # test_1_agent_1_world__no_server([1,1,2])
    # test_1_agent_1_world__no_server([1,1,3])
    
    # test_1_agent_1_world__no_server([1,2,1])
    # test_1_agent_1_world__no_server([1,2,2])
    # test_1_agent_1_world__no_server([1,2,3])
    
    # test_1_agent_1_world__no_server([1,3,1])
    # test_1_agent_1_world__no_server([1,3,2])
    # test_1_agent_1_world__no_server([1,3,3])
    
if args.comp_softmax:
    ids = [[1,1,1], [1,1,2], [1,1,3]]
    # ids = [[1,2,1], [1,2,2], [1,2,3]]
    # ids = [[1,3,1], [1,3,2], [1,3,3]]
    launch_client('{} -is -sql -exp -nnu --codename "singleagent.singlegame.noserver.{}.{}.{}"'.            format(id_args(ids[0]), ids[0][0],ids[0][1],ids[0][2]))
    launch_client('{} -is -sql -exp -nnu --codename "singleagent.singlegame.noserver.{}.{}.{}"'.            format(id_args(ids[1]), ids[1][0],ids[1][1],ids[1][2]))
    launch_client('{} -is -sql -exp -nnu --codename "singleagent.singlegame.noserver.{}.{}.{}"'.            format(id_args(ids[2]), ids[2][0],ids[2][1],ids[2][2]))
    launch_client('{} -is -sql -exp -nnu -sm --codename "singleagent.singlegame.noserver.{}.{}.{}.softmax"'.format(id_args(ids[0]), ids[0][0],ids[0][1],ids[0][2]))
    launch_client('{} -is -sql -exp -nnu -sm --codename "singleagent.singlegame.noserver.{}.{}.{}.softmax"'.format(id_args(ids[1]), ids[1][0],ids[1][1],ids[1][2]))
    launch_client('{} -is -sql -exp -nnu -sm --codename "singleagent.singlegame.noserver.{}.{}.{}.softmax"'.format(id_args(ids[2]), ids[2][0],ids[2][1],ids[2][2]))

