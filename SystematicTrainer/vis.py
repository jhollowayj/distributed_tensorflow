import matplotlib.pyplot as plt
import numpy as np
import math

plt.ion()
show_bar_once = True
fig, ax = plt.subplots(figsize=(13, 12))
print "=-=-=-=-=-=- plt.ion set -=-=-=-=-=-="
def draw_matrix_w_arrows(max_vals, dirs, difs):
    global show_bar_once
    s = ax.imshow(max_vals, interpolation='none')
    clear_arrows()
    for x in range(10):
        for y in range(10):
            _draw_arrows(x,y, dirs[x][y], difs[x][y])
    if show_bar_once:
        plt.colorbar(s)
        show_bar_once = not show_bar_once
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

arrows = []
def clear_arrows():
    ''' axes annotations must be removed, or else they persist.'''
    global arrows
    for a in arrows:
        a.remove()
    arrows = []

def __calc_new_len_from_diffs(diff):
    # return np.max(0.2, np.min(diff * 4, 0.9))
    lenmax = 0.95
    lenmin = 0.2
    multiplyer = 2.0
    
    res = ((diff * multiplyer) * (lenmax - lenmin)) + lenmin
    return max(0, min(res, 1.2)) # 0.2 <= len*4 <= 0.9

def _draw_arrows(y, x, dir, length=0.6): # x and y are actually backwards...
    total_length = __calc_new_len_from_diffs(length)
    
    # Calcs to center the arrow in the cell
    head_length = 0.1
    dstart = (total_length/2) - head_length
    dend = total_length/2

    global arrows
    if   dir == 0: arrows.append(ax.arrow(x, y+dstart, 0, -dend, head_width=0.1, head_length=head_length))
    elif dir == 1: arrows.append(ax.arrow(x-dstart, y, +dend, 0, head_width=0.1, head_length=head_length))
    elif dir == 2: arrows.append(ax.arrow(x, y-dstart, 0, +dend, head_width=0.1, head_length=head_length))
    elif dir == 3: arrows.append(ax.arrow(x+dstart, y, -dend, 0, head_width=0.1, head_length=head_length))
    else: pass #(Wall/end)