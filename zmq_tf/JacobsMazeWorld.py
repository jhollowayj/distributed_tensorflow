from interface import World
import numpy as np
from enum import Enum
import numpy as np
from enum import Enum


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
    def __init__(self, world_id=1, task_id = 1, agent_id = 1):
        self.agent_id = agent_id
        self.task_id = task_id
        self.world_id = world_id
        self.action_mapping = get_agent_mapping(self.agent_id)

        # Define start and end points
        self.startLocation = self.getStartPoint(self.task_id)
        self.endLocation = self.getGoalPoint(self.task_id)
        self.restart()
        
    def restart(self):
        self.target = self.endLocation
        self.agent_location = self.startLocation
        self.currentScore = 0.0
        self.time = 0
        self.did_finish = False
        self.maze = self.buildMaze()

    def buildMaze(self):
        return {
            1: self.buildMaze_staggered,
            2: self.buildMaze_orgMaze,
            3: self.buildMaze_upRight,
        }[self.world_id]() # Pick the right function and call it.
        
    def heatmap_adder(self):
        arr = np.zeros(self.maze.shape)
        arr[self.agent_location[0], self.agent_location[1]] = 1
        return arr
    def getStartPoint(self, taskId):
        return {
            1: ( 5,  5),
            2: (10,  1),
            3: ( 1, 10)
        }[taskId]
    def getGoalPoint(self, taskId):
        return {
            1: ( 1,  1),
            2: ( 1, 10),
            3: ( 3, 3)
        }[taskId]

    #######################################################################
    def start(self): # if nneded gpu, allocate, etc.
        return

    def is_running(self):
        return not self.did_finish

    def render(self):
        mazecopy = []
        for x in np.array(self.maze):
            t = []
            for y in x:
                if y.value == 1:
                    t.append(' ')
                else: 
                    t.append(str(y.value))
            mazecopy.append(t) 
        # mazecopy[self.agent_location[0], self.agent_location[1]] = -1
        print np.array(mazecopy)
        # print "                              Score:{}  (+ {})".format(m.get_score(), score)

    def checkEnd(self, desiredCell):
        return self.endLocation == desiredCell
        
        # return desiredCell == (10, 10) or \ # TESTING CODE FOR STAGGER WORLD
        #        desiredCell == ( 1,  1) or \
        #        desiredCell == (10,  1) or \
        #        desiredCell == ( 1, 10)
        
    def act(self, action, cur_x=None, cur_y=None):
        if cur_x is None: cur_x = self.agent_location[0]
        if cur_y is None: cur_y = self.agent_location[1]

        self.time += 1
        # print action
        move = self.action_mapping[action]

        reward_for_movement = 0.0
        
        # Calculate that cell
        desiredCell = None
        if   move == direction.n: desiredCell = (cur_x-1, cur_y)
        elif move == direction.s: desiredCell = (cur_x+1, cur_y)
        elif move == direction.e: desiredCell = (cur_x, cur_y+1)
        elif move == direction.w: desiredCell = (cur_x, cur_y-1)
        else: print "Error on calculating the desired cell, action was invalid..."
        desiredCellType = self.maze[desiredCell[0], desiredCell[1]]
        
        # Calcualte Reward + Move
        terminal = False
        if desiredCellType == maze_object.wall:   # Was it a wall?  I don't like walls...
            reward_for_movement -= 0.1
            self.agent_location = (cur_x, cur_y)
            # 1 # Do nothing
        elif self.checkEnd(desiredCell):         # Did I win?
            reward_for_movement = 10.0
            self.agent_location = desiredCell
            terminal = True
            self.did_finish = True
        elif desiredCellType == maze_object.coin:   # No?  Is there a coin?
            self.agent_location = desiredCell
            # reward_for_movement += 1
            # self.maze[desiredCell[0]][desiredCell[1]] = maze_object.open
        elif desiredCellType == maze_object.open:   # No?  Can I go there?
            self.agent_location = desiredCell
            # reward_for_movement += 1

        self.currentScore += reward_for_movement

        # next_state = (None, None) if terminal else self.agent_location
        next_state = self.agent_location
        return next_state, reward_for_movement, terminal

    def get_time(self):
        return self.time

    def calc_distance_to_goal(self, cur_x=None, cur_y=None):
        if cur_x is None: cur_x = self.agent_location[0]
        if cur_y is None: cur_y = self.agent_location[1]
        dx = np.abs(self.cur_x - self.endLocation[0])
        dy = np.abs(self.cur_y - self.endLocation[1])
        return dx, dy
        
    def get_surrounding_square(self, cur_x=None, cur_y=None):
        if cur_x is None: cur_x = self.agent_location[0]
        if cur_y is None: cur_y = self.agent_location[1]
        arr = []
        ### N S W E ###
        arr.append(self.maze[cur_x-1, cur_y  ])
        arr.append(self.maze[cur_x+1, cur_y  ])
        arr.append(self.maze[cur_x,   cur_y-1])
        arr.append(self.maze[cur_x,   cur_y+1])
        ### Diagonals ###
        # arr.append(self.maze[self.agent_location[0]-1, self.agent_location[1]-1])
        # arr.append(self.maze[self.agent_location[0]-1, self.agent_location[1]+1])
        # arr.append(self.maze[self.agent_location[0], self.agent_location[1]  ])
        # arr.append(self.maze[self.agent_location[0]+1, self.agent_location[1]-1])
        # arr.append(self.maze[self.agent_location[0]+1, self.agent_location[1]+1])
        return arr

    give_expanded_space = True
    def get_state(self, cur_x=None, cur_y=None):
        if cur_x is None: cur_x = self.agent_location[0]
        if cur_y is None: cur_y = self.agent_location[1]
        if self.give_expanded_space:
            arr = []
            arr += [cur_x, cur_y]
            # arr += self.get_surrounding_square(cur_x, cur_y) # Sync these with get_state__maxes
            # arr += self.calc_distance_to_goal()
            return arr
        else:
            return self.agent_location # For now, we just return the (x,y) position.
            
    def get_state__maxes(self):
        if self.give_expanded_space:
            arr = []
            arr += [11, 11] # Agent location
            # arr += [10,10,10,10] # Surrounding Square # I think 10 is an ok value... it should probably be the max of maze_object options though.
            # arr += [11,11] # self.calc_distance_to_goal()
            return arr
        else:
            return [11,11] # For now, we just return the (x,y) position.
            
    def get_all_possible_states(self):
        states = []
        for x in range(12):
            for y in range(12):
                if self.maze[x, y] is not maze_object.wall:
                    states.append(self.get_state(x, y))
        return states
        
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
    
            
    def buildMaze_staggered(self):
        w, o, c = maze_object.wall, maze_object.open, maze_object.coin
        return np.array([
            [w,w,w,w,w,w,w,w,w,w,w,w], # [w w w w w w w w w w w w]
            [w,c,c,c,w,c,c,c,w,c,c,w], # [w       w       w     w]
            [w,c,w,c,c,c,w,c,c,c,w,w], # [w   w       w       w w]
            [w,c,c,c,w,c,c,c,w,c,c,w], # [w       w       w     w]
            [w,c,w,c,c,c,w,c,c,c,w,w], # [w   w       w       w w]
            [w,c,c,c,w,c,c,c,w,c,c,w], # [w       w       w     w]
            [w,c,w,c,c,c,w,c,c,c,w,w], # [w   w       w       w w]
            [w,c,c,c,w,c,c,c,w,c,c,w], # [w       w       w     w]
            [w,c,w,c,c,c,w,c,c,c,w,w], # [w   w       w       w w]
            [w,c,c,c,w,c,c,c,w,c,c,w], # [w       w       w     w]
            [w,c,w,c,c,c,w,c,c,c,w,w], # [w   w       w       w w]
            [w,w,w,w,w,w,w,w,w,w,w,w]  # [w w w w w w w w w w w w]
        ])
        
    def buildMaze_upRight(self):
        w, o, c = maze_object.wall, maze_object.open, maze_object.coin
        return np.array([
            [w,w,w,w,w,w,w,w,w,w,w,w],  # [w w w w w w w w w w w w]
            [w,c,c,c,c,c,c,c,c,c,c,w],  # [w                     w]
            [w,c,w,w,w,w,w,w,w,w,w,w],  # [w   w w w w w w w w w w]
            [w,c,w,w,w,w,w,w,w,w,w,w],  # [w   w w w w w w w w w w]
            [w,c,w,w,w,w,w,w,w,w,w,w],  # [w   w w w w w w w w w w]
            [w,c,w,w,w,w,w,w,w,w,w,w],  # [w   w w w w w w w w w w]
            [w,c,w,w,w,w,w,w,w,w,w,w],  # [w   w w w w w w w w w w]
            [w,c,w,w,w,w,w,w,w,w,w,w],  # [w   w w w w w w w w w w]
            [w,c,w,w,w,w,w,w,w,w,w,w],  # [w   w w w w w w w w w w]
            [w,c,w,w,w,w,w,w,w,w,w,w],  # [w   w w w w w w w w w w]
            [w,c,w,w,w,w,w,w,w,w,w,w],  # [w   w w w w w w w w w w]
            [w,w,w,w,w,w,w,w,w,w,w,w]   # [w w w w w w w w w w w w]
        ])
        maze[self.endLocation[0], self.endLocation[1]] = maze_object.exit

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
            [w,o,o,o,w,o,w,w,w,w,o,w], # [w       w   w w w w   w],
            [w,o,w,w,w,o,w,o,o,o,o,w], # [w   w w w   w         w],
            [w,o,w,o,o,o,o,o,w,o,o,w], # [w   w           w     w],
            [w,w,w,w,w,w,w,w,w,w,w,w]  # [w w w w w w w w w w w w]
        ])

    #######################################################################
debugging = False
if debugging:    
    m = JacobsMazeWorld()
    print m.get_state()
    for _ in range(10):
        score = m.act(0) # go north my son.
        m.render()
        
    print "======================================================================="

    for _ in range(9):
        score = m.act(1) # straight home to the morning light!
        m.render()
    print m.get_state()
    