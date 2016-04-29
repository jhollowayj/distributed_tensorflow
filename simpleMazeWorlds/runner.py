import dqn_with_gym
import JacobsMazeWorld
import numpy as np
import time

num_episodes = 20000
pleaseRender = False

world = JacobsMazeWorld.JacobsMazeWorld() # use default world params

agent = dqn_with_gym.Agent(
    state_size=world.get_state_space(),
    number_of_actions=len(world.get_action_space()),
    memory=100000,
    epsilon=1.0/20.0,
    batch_size=5,
    anealing_size=600)

runId = "20t_ProbE_Rp100n10"
csv = open('results_{}.csv'.format(runId),'w', 0)
csv.write("episode,total_reward,mean_cost,max_q,endEpsilon,didFinish\n")
starttime = time.time()
for e in xrange(num_episodes):
    world.reset()
    done = False
    agent.new_episode()
    total_cost = 0.0
    frame = 0
    max_q = 0 - np.Infinity
    arr = world.heatmap_adder()
    actions = [0,0,0,0]
    while world.is_running() and world.get_time() < 200: 
        frame += 1
        cur_state = world.get_state()
        action, values = agent.select_action(np.array(cur_state))
        actions[action] += 1
        max_q = max(max_q, np.max(values))
        reward = world.act(action)
        total_cost += agent.train(reward) # IS this where it goes wrong?
        if pleaseRender:
            world.render()
        arr += world.heatmap_adder()
    # REPORTTING
    runtime = time.time() - starttime
    totaltime = runtime / (e+1) * num_episodes
    print
    print arr
    # print agent.getRewardsPerSquare(maze=world.mazeMask())
    # a1 = agent.getRewardsPerSquare(0,maze=world.mazeMask())
    # a2 = agent.getRewardsPerSquare(1,maze=world.mazeMask())
    # # print a1
    # # print a2
    # print a1-a2
    

    print "\r episode: %6d::%4d/%4ds::    total reward: %6d    mean cost: %13.9f,   max_q: %10.6f,  endEpsilon: %4.3f,   didFinish: %s" % \
          (e, runtime, totaltime, world.get_score(), (total_cost/frame),
           max_q, agent.calculate_epsilon(), "No" if world.is_running() else "Yes") 

    csv.write("{},{},{},{},{},{}\n".format(e, world.get_score(), (total_cost/frame), max_q, agent.calculate_epsilon(), 0 if world.is_running() else 1))

csv.close()

# Investigate, world is giving xy pairs, they are right.
# Maybe add in the 3x3 squares around him?
# Try to weight the well rewarded actions more in batches 
