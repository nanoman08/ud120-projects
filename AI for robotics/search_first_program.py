# -*- coding: utf-8 -*-
"""
Created on Tue Nov 01 14:59:31 2016

@author: CHOU_H
"""


# ----------
# User Instructions:
# 
# Define a function, search() that returns a list
# in the form of [optimal path length, row, col]. For
# the grid shown below, your function should output
# [11, 4, 5].
#
# If there is no valid path from the start point
# to the goal, your function should return the string
# 'fail'
# ----------

# Grid format:
#   0 = Navigable space
#   1 = Occupied space
import copy
grid = [[0, 0, 1, 0, 0, 0],
        [0, 0, 1, 0, 1, 0],
        [0, 1, 1, 0, 1, 0],
        [0, 1, 0, 0, 1, 0],
        [0, 0, 0, 1, 1, 0]]
init = [0, 0]
goal = [len(grid)-1, len(grid[0])-1]
cost = 1

delta = [[-1, 0], # go up
         [ 0,-1], # go left
         [ 1, 0], # go down
         [ 0, 1]] # go right

delta_name = ['^', '<', 'v', '>']

def search(grid,init,goal,cost):
    # ----------------------------------------
    # insert code here

    if grid[init[0]][init[1]] == 1:
        raise ValueError, "Robot starts on the forbidden region"
    grid_and_value = grid
    for i in range(len(grid)):
        for j in range(len(grid[0])):
            grid_and_value[i][j] = [grid[i][j], 0]
    
    # assign initial occupation to 2 (this is to differentiate from forbidden state = 1), 
    # assign initial occupation value to 0
    value = 0    
    grid_and_value[init[0]][init[1]] = [2, value]    
    
    temp = [[value, init[0], init[1]]]
    found = False
    resign = False
    while not found and not resign:
        temp_2 = []
        value += cost
        for j in range(len(temp)):
            [c_value, c_x, c_y] = temp[j]
            for [a, b] in [[-1,0],[0,-1],[1,0],[0,1]]:
            
                if c_x + a in range(len(grid)) and c_y + b in range(len(grid[0])):
                    if grid_and_value[c_x+a][c_y+b][0] not in [1,2]:
                        grid_and_value[c_x+a][c_y+b][0] = 2
                        grid_and_value[c_x+a][c_y+b][1] = value
                        temp_2.append([value, c_x+a, c_y+b])
                        if [c_x+a, c_y+b] == goal:
                            found = True
                            final = [value, c_x+a, c_y+b]
        
      
        if temp_2 == []:
            resign = True
            final = 'fail'
            
        temp = temp_2
    
    return final


                    
print search(grid,init,goal,cost)                  
            
                
                
# -----------
# User Instructions:
# 
# Modify the function search so that it returns
# a table of values called expand. This table
# will keep track of which step each node was
# expanded.
#
# Make sure that the initial cell in the grid 
# you return has the value 0.
# ----------

grid = [[0, 0, 1, 0, 0, 0],
        [0, 0, 0, 0, 0, 0],
        [0, 0, 1, 0, 1, 0],
        [0, 0, 1, 0, 1, 0],
        [0, 0, 1, 0, 1, 0]]
        
grid = [[0, 0, 1, 0, 0, 0],
        [0, 0, 0, 0, 0, 0],
        [0, 0, 1, 0, 1, 0],
        [0, 0, 1, 0, 1, 0],
        [0, 0, 1, 0, 1, 0]]
        
init = [0, 0]
goal = [len(grid)-1, len(grid[0])-1]
cost = 1

delta = [[-1, 0], # go up
         [ 0,-1], # go left
         [ 1, 0], # go down
         [ 0, 1]] # go right

delta_name = ['^', '<', 'v', '>']

def search2(grid,init,goal,cost):
    # ----------------------------------------
    # modify code below
    # ----------------------------------------
    closed = [[0 for row in range(len(grid[0]))] for col in range(len(grid))]
    closed[init[0]][init[1]] = 1
    expand = [[-1 for row in range(len(grid[0]))] for col in range(len(grid))]
    x = init[0]
    y = init[1]
    g = 0

    open = [[g, x, y]]


    found = False  # flag that is set when search is complete
    resign = False # flag set if we can't find expand
    count = 0
    while not found and not resign:
        if len(open) == 0:
            resign = True
        else:
            open.sort()
            open.reverse()
            next = open.pop()
            x = next[1]
            y = next[2]
            g = next[0]
            expand[x][y]=count
            count +=1
            
            if x == goal[0] and y == goal[1]:
                found = True
            else:
                for i in range(len(delta)):
                    x2 = x + delta[i][0]
                    y2 = y + delta[i][1]
                    if x2 >= 0 and x2 < len(grid) and y2 >=0 and y2 < len(grid[0]):
                        if closed[x2][y2] == 0 and grid[x2][y2] == 0:
                            g2 = g + cost
                            open.append([g2, x2, y2])
                            closed[x2][y2] = g2
                

                     
 
                

            

    return expand

expand = search2(grid,init,goal,cost)
for i in range(len(expand)):
    print expand[i]
        
        
 
        
        
    
    
               


