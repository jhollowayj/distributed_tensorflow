from interface import World
import numpy as np
from enum import Enum
import numpy as np
from enum import Enum
import cv2
import time

class direction(Enum):
    n = 0
    e = 1
    s = 2
    w = 3
class maze_object(Enum):
    user = 0
    wall = 1
    open = 4
    coin = 7
    exit = 10
    # visited = 4 #Maybe add this in later?
    
# def agent1Mapping(): return [direction.n, direction.e, direction.w]
def agent1Mapping(): return [direction.n, direction.e, direction.s, direction.w]
def agent2Mapping(): return [direction.e, direction.s, direction.w, direction.n]
def agent3Mapping(): return [direction.s, direction.w, direction.n, direction.e]
def get_agent_mapping(id):
    return {
        1: agent1Mapping,
        2: agent2Mapping,
        3: agent3Mapping
    }[id]()

class JacobsMazeWorld(World):
    def __init__(self, world_id=1, task_id = 1, agent_id = 1, random_start = False, onehot_state=False):
        self.agent_id = agent_id
        self.task_id = task_id
        self.world_id = world_id
        self.random_start = random_start
        self.action_mapping = get_agent_mapping(self.agent_id)
        self.onehot_state = onehot_state
        self.rewards = {
            maze_object.exit: 10.0,
            maze_object.coin:  1.0,
            maze_object.wall: -0.1,
            maze_object.open:  0.0,
            maze_object.user:  0.0,
        }
        self.render_to_console = False
        self.render_to_cv2 = True
        self.window_name = None
        self.endLocation = self.getGoalPoint(self.task_id)
        self.restart()
        
        
    def restart(self):
        self.maze = self.buildMaze()
        self.currentScore = 0.0
        self.time = 0
        self.did_finish = False
        self.agent_location = self.get_valid_rand_start_cell() \
                              if self.random_start \
                              else self.getStartPoint(self.task_id)

    def buildMaze(self):
        maze = {
            1: self.buildMaze_staggered,
            2: self.buildMaze_orgMaze,
            3: self.buildMaze_upRight,
        }[self.world_id]() # Pick the right function and call it.
        maze[self.endLocation[0], self.endLocation[1]] = maze_object.exit
        return maze
        
    def getStartPoint(self, taskId):
        return {
            1: ( 1,  1),
            2: ( 1, 10),
            3: (10,  1)
        }[taskId]
    def getGoalPoint(self, taskId):
        return {
            1: ( 5, 5),
            2: ( 8, 8),
            3: ( 3, 3)
        }[taskId]

    def get_valid_rand_start_cell(self):
        (x, y) = np.random.randint(12, size=2)
        if self.maze[x][y] == maze_object.wall or self.maze[x][y] == maze_object.exit:
            (x, y) = self.get_valid_rand_start_cell() # go until valid
        return (x, y)
        
    #######################################################################
    def start(self): # if nneded gpu, allocate, etc.
        return

    def is_running(self):
        return not self.did_finish
        
    def act(self, action, cur_x=None, cur_y=None):
        if cur_x is None: cur_x = self.agent_location[0]
        if cur_y is None: cur_y = self.agent_location[1]

        self.time += 1
        terminal = False
        
        # Calculate that cell
        desiredCell = None
        move = self.action_mapping[action]
        if   move == direction.n: desiredCell = (cur_x-1, cur_y)
        elif move == direction.s: desiredCell = (cur_x+1, cur_y)
        elif move == direction.e: desiredCell = (cur_x, cur_y+1)
        elif move == direction.w: desiredCell = (cur_x, cur_y-1)
        else: print "Error on calculating the desired cell, action was invalid..."
        
        desiredCellType = self.maze[desiredCell[0], desiredCell[1]]
        reward_for_movement = self.rewards[desiredCellType] # get your reward
        self.currentScore += reward_for_movement 

        if desiredCellType != maze_object.wall: # move if not a wall.
            self.agent_location = desiredCell
        else: # This might need to be added, but if someone just wants to check something,
            self.agent_location = (cur_x, cur_y) #  ... it might break that...
                                                 #  Sorry it's spaghetti code... :(

        # Terminal?
        if self.endLocation == desiredCell:
            terminal = True
            self.did_finish = True

        next_state = self.get_state()
        return next_state, reward_for_movement, terminal

    def get_time(self):
        return self.time

    def create_onehot_state(self, x, y):
        a = np.zeros(12*12)
        a[ (x*12) + y ] = 1
        return a
        
    def get_state(self, cur_x=None, cur_y=None):
        if cur_x is None: cur_x = self.agent_location[0]
        if cur_y is None: cur_y = self.agent_location[1]
        if self.onehot_state:
           return self.create_onehot_state(cur_x, cur_y)
        else:
            return [cur_x, cur_y]
            
    def get_state_xy(self, cur_x=None, cur_y=None):
        if cur_x is None: cur_x = self.agent_location[0]
        if cur_y is None: cur_y = self.agent_location[1]
        return [cur_x, cur_y]
            
    def get_state__maxes(self):
        if self.onehot_state:
            return np.ones(12*12)
        else:
            return [11,11] # For now, we just return the (x,y) position.
            
    def get_score(self):
        return self.currentScore

    def get_action_space(self):
        return range(len(self.action_mapping))

    def get_state_space(self):
        return (len(self.get_state()),) # Number of return items 

    def reset(self):
        self.restart()

    def load(self):
        return # Do nothing for now

    #######################################################################

    def get_all_possible_states(self):
        states = []
        for x in range(12):
            for y in range(12):
                if self.maze[x, y] is not maze_object.wall:
                    states.append([x, y])
        return states
    
    def heatmap_adder(self):
        arr = np.zeros(self.maze.shape)
        arr[self.agent_location[0], self.agent_location[1]] = 1
        return arr

    def render(self):
        if self.render_to_console:
            mazeCopy = []
            for x in np.array(self.maze):
                t = []
                for y in x:
                    if y.value == 1:
                        t.append(' ')
                    else: 
                        t.append(str(y.value))
                mazeCopy.append(t) 
            mazeCopy[self.agent_location[0], self.agent_location[1]] = -1
            print np.array(mazeCopy)
            print "                              Score:{}  (+ {})".format(m.get_score(), score)
        if self.render_to_cv2:
            if self.window_name is None:
                self.window_name = "maze.{}.{}.{}".format(self.world_id, self.task_id, self.agent_id) 
                cv2.startWindowThread()
                cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
            cp = np.empty([12, 12, 3], dtype=float)
            for i in range(12):
                for j in range(12):
                    cp[i,j] = [self.maze[i,j].value] * 3
            cp += np.min(cp) # Scale to 0. -> 1.
            cp = cp / np.max(cp)
            cp[self.agent_location[0], self.agent_location[1]] = [maze_object.user.value] * 3

            cv2.imshow(self.window_name, cp)
            time.sleep(0.01)

    #######################################################################

    def buildMaze_staggered(self):
        w, o = maze_object.wall, maze_object.open
        return np.array([
            [w,w,w,w,w,w,w,w,w,w,w,w], # [w w w w w w w w w w w w]
            [w,o,o,o,w,o,o,o,w,o,o,w], # [w       w       w     w]
            [w,o,w,o,o,o,w,o,o,o,w,w], # [w   w       w       w w]
            [w,o,o,o,w,o,o,o,w,o,o,w], # [w       w       w     w]
            [w,o,w,o,o,o,w,o,o,o,w,w], # [w   w       w       w w]
            [w,o,o,o,w,o,o,o,w,o,o,w], # [w       w       w     w]
            [w,o,w,o,o,o,w,o,o,o,w,w], # [w   w       w       w w]
            [w,o,o,o,w,o,o,o,w,o,o,w], # [w       w       w     w]
            [w,o,w,o,o,o,w,o,o,o,w,w], # [w   w       w       w w]
            [w,o,o,o,w,o,o,o,w,o,o,w], # [w       w       w     w]
            [w,o,w,o,o,o,w,o,o,o,w,w], # [w   w       w       w w]
            [w,w,w,w,w,w,w,w,w,w,w,w]  # [w w w w w w w w w w w w]
        ])
        
    def buildMaze_orgMaze(self):
        w, o, c = maze_object.wall, maze_object.open, maze_object.coin
        return np.array([
            [w,w,w,w,w,w,w,w,w,w,w,w], # [w w w w w w w w w w w w],
            [w,o,o,o,o,o,o,o,o,o,o,w], # [w                     w],
            [w,o,w,w,w,w,w,w,w,w,o,w], # [w   w w w w w w w w   w],
            [w,o,w,o,o,o,w,o,o,o,o,w], # [w   w       w         w],
            [w,o,w,o,w,o,w,o,w,w,o,w], # [w   w   w   w   w w   w],
            [w,o,w,o,w,o,w,o,w,o,o,w], # [w   w   w   w   w     w],
            [w,o,w,o,w,o,o,o,w,o,w,w], # [w   w   w       w   w w],
            [w,o,w,o,w,o,o,w,o,o,o,w], # [w   w   w     w       w],
            [w,o,o,o,w,o,w,w,o,w,o,w], # [w       w   w w   w   w],
            [w,o,w,w,w,o,w,o,o,w,o,w], # [w   w w w   w     w   w],
            [w,o,w,o,o,o,o,o,w,o,o,w], # [w   w           w     w],
            [w,w,w,w,w,w,w,w,w,w,w,w]  # [w w w w w w w w w w w w]
        ])

    def buildMaze_upRight(self):
        w, o, c = maze_object.wall, maze_object.open, maze_object.coin
        return np.array([
            [w,w,w,w,w,w,w,w,w,w,w,w],  # [w w w w w w w w w w w w]
            [w,o,w,o,o,o,o,o,o,o,o,w],  # [w   w                 w]
            [w,o,w,w,w,o,w,w,w,w,w,w],  # [w   w w w   w w w w w w]
            [w,o,w,o,o,o,o,o,o,w,o,w],  # [w   w             w   w]
            [w,o,o,o,w,w,w,w,o,w,o,w],  # [w       w w w w   w   w]
            [w,o,w,o,w,o,o,o,o,o,o,w],  # [w   w   w             w]
            [w,o,o,o,o,o,o,w,o,w,o,w],  # [w             w   w   w]
            [w,o,w,o,w,w,w,w,o,o,o,w],  # [w   w   w w w w       w]
            [w,o,w,o,o,o,o,o,o,w,o,w],  # [w   w             w   w]
            [w,w,w,w,w,w,o,w,w,w,o,w],  # [w w w w w w   w w w   w]
            [w,o,o,o,o,o,o,o,o,w,o,w],  # [w                 w   w]
            [w,w,w,w,w,w,w,w,w,w,w,w]   # [w w w w w w w w w w w w]
        ])

    #######################################################################
if __name__ == "__main__":
    mazeWorld = JacobsMazeWorld()
    print mazeWorld.get_state()
    for _ in range(10):
        score = mazeWorld.act(0) # go north my son.
        mazeWorld.render()
        
    print "======================================================================="

    for _ in range(9):
        score = mazeWorld.act(1) # straight home to the morning light!
        mazeWorld.render()
    print mazeWorld.get_state()
    