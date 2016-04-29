from interface import World
import numpy as np
from enum import Enum


class direction(Enum):
    s = 0
    w = 1
    n = 2
    e = 3
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
        

class JacobsMazeWorld(World):
    def __init__(self, task_id = 2, action_mapping=agent3Mapping()):
        self.action_mapping = action_mapping
        self.task_id = task_id

        # Define start and end points
        self.startLocation = self.getStartPoint(self.task_id)
        self.endLocation = self.getGoalPoint(self.task_id)
        self.restart()
        
    def restart(self):
        self.target = self.endLocation
        self.agent_location = self.startLocation
        self.currentScore = 0
        self.time = 0
        self.did_finish = False
        self.maze = self.buildMaze()
        
    def buildMaze(self):
        w = maze_object.wall
        o = maze_object.open
        c = maze_object.coin
        maze = np.array([               # What this one looks like
            [w,w,w,w,w,w,w,w,w,w,w,w],  # 1 1 1 1 1 1 1 1 1 1 1 1 | 0
            [w,c,c,c,w,c,c,c,w,c,w,w],  # 1     1             e 1 | 1
            [w,c,w,c,c,c,w,c,c,c,w,w],  # 1   1   1 1 1 1 1 1   1 | 2
            [w,c,c,c,w,c,c,c,w,c,c,w],  # 1   1   1   1         1 | 3
            [w,c,w,c,c,c,w,c,c,c,w,w],  # 1   1   1   1   1 1   1 | 4
            [w,c,c,c,w,c,c,c,w,c,c,w],  # 1   1   1   1   1     1 | 5
            [w,c,w,c,c,c,w,c,c,c,w,w],  # 1   1   1       1     1 | 6
            [w,c,c,c,w,c,c,c,w,c,c,w],  # 1   1   1       1     1 | 7
            [w,c,w,c,c,c,w,c,c,c,w,w],  # 1   1   1   1 1 1 1   1 | 8
            [w,c,c,c,w,c,c,c,w,c,c,w],  # 1   1 1 1   1         1 | 9
            [w,c,w,c,c,c,w,c,c,c,w,w],  # 1 s 1   1       1     1 | 0
            [w,w,w,w,w,w,w,w,w,w,w,w]   # 1 1 1 1 1 1 1 1 1 1 1 1 | 1
        ])                              # ------------------------+
                                        # 0 1 2 3 4 5 6 7 8 9 0 1
        # maze = np.ones((12,12))
        # maze[0:12,0:12] = maze_object.wall
        # maze[1:11,1:11] = maze_object.coin
        # print maze
        maze[self.endLocation[0], self.endLocation[1]] = maze_object.exit
        return maze
    def mazeMask(self):
        mask = np.array([
            [1,1,1,1,1,1,1,1,1,1,1,1], 
            [1,1,1,0,0,0,0,0,0,0,0,1], 
            [1,1,1,0,1,1,1,1,1,1,1,1], 
            [1,1,1,0,1,1,1,1,1,1,1,1], 
            [1,1,1,0,1,1,1,1,1,1,1,1], 
            [1,1,1,0,1,1,1,1,1,1,1,1], 
            [1,1,1,0,1,1,1,1,1,1,1,1], 
            [1,1,1,0,1,1,1,1,1,1,1,1], 
            [1,0,0,0,1,1,1,1,1,1,1,1], 
            [1,0,1,1,1,1,1,1,1,1,1,1], 
            [1,0,1,1,1,1,1,1,1,1,1,1], 
            [1,1,1,1,1,1,1,1,1,1,1,1]  
        ])
        mask -= 1
        mask *= -1
        return mask

    def heatmap_adder(self):
        arr = np.zeros(self.maze.shape)
        arr[self.agent_location[0], self.agent_location[1]] = 1
        return arr
    def getStartPoint(self, taskId):
        return {
            1: ( 1,  1),
            2: ( 6, 6),
            3: ( 1, 10)
        }[taskId]
    def getGoalPoint(self, taskId):
        return {
            1: (10, 10),
            2: ( 1,  1),
            3: ( 3, 3)
        }[taskId]

    #######################################################################
    def start(self): # if nneded gpu, allocate, etc.
        return

    def is_running(self):
        return not self.did_finish

    def render(self):
        mazecopy = np.array(self.maze)
        mazecopy[self.agent_location[0], self.agent_location[1]] = maze_object.user
        print mazecopy
        print "                              Score:{}  (+ {})".format(m.get_score(), score)

    def checkEnd(self, desiredCell):
        # return self.endLocation == desiredCell
        return desiredCell == (10, 10) or \
               desiredCell == ( 1,  1) or \
               desiredCell == (10,  1) or \
               desiredCell == ( 1, 10)
    def act(self, action):
        self.time += 1
        # print action
        move = self.action_mapping[action]

        reward_for_movement = 0
        
        # Calculate that cell
        desiredCell = None
        if   move == direction.n: desiredCell = (self.agent_location[0]-1, self.agent_location[1])
        elif move == direction.s: desiredCell = (self.agent_location[0]+1, self.agent_location[1])
        elif move == direction.e: desiredCell = (self.agent_location[0], self.agent_location[1]+1)
        elif move == direction.w: desiredCell = (self.agent_location[0], self.agent_location[1]-1)
        else: print "Error on calculating the desired cell, action was invalid..."
        desiredCellType = self.maze[desiredCell[0], desiredCell[1]]
        
        # Calcualte Reward + Move
        if desiredCellType == maze_object.wall:   # No?  Was it a wall?  I don't like walls...
            reward_for_movement -= 10
            # 1 # Do nothing
        elif self.checkEnd(desiredCell):         # Did I win?
            reward_for_movement = 500
            self.agent_location = desiredCell
            self.did_finish = True
        elif desiredCellType == maze_object.coin:   # No?  Is there a coin?
            self.agent_location = desiredCell
            reward_for_movement += 1
            self.maze[desiredCell[0]][desiredCell[1]] = maze_object.open
        elif desiredCellType == maze_object.open:   # No?  Can I go there?
            self.agent_location = desiredCell
            # reward_for_movement += 1

        self.currentScore += reward_for_movement

        return reward_for_movement

    def get_time(self):
        return self.time

    def calc_distance_to_goal(self):
        dx = np.abs(self.agent_location[0] - self.endLocation[0])
        dy = np.abs(self.agent_location[1] - self.endLocation[1])
        return dx, dy
        
    def get_surrounding_square(self):
        arr = []
        # arr.append(self.maze[self.agent_location[0]-1, self.agent_location[1]-1])
        arr.append(self.maze[self.agent_location[0]-1, self.agent_location[1]  ])
        # arr.append(self.maze[self.agent_location[0]-1, self.agent_location[1]+1])
        arr.append(self.maze[self.agent_location[0], self.agent_location[1]-1])
        # arr.append(self.maze[self.agent_location[0], self.agent_location[1]  ])
        arr.append(self.maze[self.agent_location[0], self.agent_location[1]+1])
        # arr.append(self.maze[self.agent_location[0]+1, self.agent_location[1]-1])
        arr.append(self.maze[self.agent_location[0]+1, self.agent_location[1]  ])
        # arr.append(self.maze[self.agent_location[0]+1, self.agent_location[1]+1])
        return arr
    
    def get_state(self):
        give_expanded_space = True
        if give_expanded_space:
            arr = []
            arr += self.agent_location
            arr += self.get_surrounding_square()
            arr += self.calc_distance_to_goal()
            return arr
        else:
            return self.agent_location # For now, we just return the (x,y) position.
        
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
