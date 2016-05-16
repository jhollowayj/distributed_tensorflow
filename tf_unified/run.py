import Runner, DQN, statistics
import threading, time

num_runners = 10
runners = []
num_episodes = 6000
runner_configs = [[0,0,0]]*num_runners
# runner_configs = [
#     [0,0,0], [0,0,1], [0,0,2],
#     [0,1,0], [0,1,1], [0,1,2],
#     [0,2,0], [0,2,1], [0,2,2],
# ]

parallel_learning_session_uuid = statistics.get_new_uuid()
global_DQN = DQN.DQN()

# TODO breakout into multithread/multiprocess

class Worker(threading.Thread):
    def __init__(self, id):
        threading.Thread.__init__(self)
        self.playon = True # A flag to notify the thread that it should finish up and exit
        self.id = id
        print "new thread of id: {}".format(self.id)

    def run(self): # is called by t.start()
        time.sleep(1)
        ids = runner_configs[self.id]
        r = Runner.Runner(
            world_id=ids[0], task_id=ids[1], agent_id=ids[2], DQN=global_DQN,
            annealing_size=50, epsilon=1.0/20.0, just_observe=False,
            num_episodes=3000, max_steps_per_episode=200, verbose=False,
            print_tag="{}.{}.{}".format(ids[0],ids[1],ids[2]),
            parallel_learning_session_uuid = parallel_learning_session_uuid,
            num_parallel_learners = num_runners)

        for e in range(num_episodes): #Meh, why not?
            if self.playon:
                r.step()
            else:
                print "Thread commanded to give up the ghost"
                break

threads = []

threads = []
print "Number of configs: {}".format(len(runner_configs))
print "configs:\n", runner_configs
for i in range(num_runners):
    t = Worker(i)
    t.daemon = True
    threads.append(t)
    t.start()

while len(threads) > 0:
    time.sleep(2)
    try:
        # Join all threads using a timeout so it doesn't block
        # Filter out threads which have been joined or are None
        for t in threads:
            if not t.isAlive():
                print "=============joining and removing thread ============", t
                t.join()
                threads.remove(t)
    except KeyboardInterrupt:
        print "Ctrl-c received! Sending kill to threads..."
        for t in threads:
            print "Clearing playon"
            t.playon = False
print "Good bye!\n\n\t\t\tThanks for playing!  :)"
