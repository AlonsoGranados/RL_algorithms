import numpy as np

grid = np.zeros((5,5))

def action_to_state(a,x,y):
    if (x == 0 and y == 1):
        return 10, 4, 1
    if (x == 0 and y == 3):
        return 5, 2, 3
    if(a == 0 and x == grid.shape[0] - 1):
        return -1, x, y
    elif(a == 1 and y == 0):
        return -1, x, y
    elif (a == 2 and x == 0):
        return -1, x, y
    elif (a == 3 and y == grid.shape[0]-1):
        return -1, x, y
    if(a == 0):
        x += 1
    elif (a == 1):
        y -= 1
    elif (a == 2):
        x -= 1
    elif (a == 3):
        y += 1
    return 0, x, y




