import matplotlib.pyplot as plt
import numpy as np

  
plt.ion()
ax = plt.axes()
print "=-=-=-=-=-=- plt.ion set -=-=-=-=-=-="
def draw_matrix(matrix):
    # print "Draw"
    plt.imshow(matrix, interpolation='none')
    # ax.arrow(0, 0, 5, 5, head_width=0.05, head_length=0.1, fc='k', ec='k')
    for x in range(10):
        for y in range(10):
            draw_arrows(x,y, matrix[x][y])
    draw_arrows(1,1, matrix[1][1])
    plt.show()
    plt.pause(0.001)
    
act = {
    -1: '#',
     0: 'n',
     1: 'e',
     2: 's',
     3: 'w',
     6: '+'
}
    
    
    
   
    
def draw_arrows(x, y, dir):
    delta = 0.04
    if   dir == 0: ax.arrow(x+delta, y, x-delta, y, head_width=0.05, head_length=0.1)# North
    elif dir == 1: ax.arrow(x, y-delta, x, y+delta, head_width=0.05, head_length=0.1)# East
    elif dir == 2: ax.arrow(x-delta, y, x+delta, y, head_width=0.05, head_length=0.1)# South
    elif dir == 3: ax.arrow(x, y+delta, x, y-delta, head_width=0.05, head_length=0.1)# West
    else: pass #(Wall/end)
