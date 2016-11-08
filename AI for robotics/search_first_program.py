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
        
# -----------
# User Instructions:
#
# Modify the the search function so that it returns
# a shortest path as follows:
# 
# [['>', 'v', ' ', ' ', ' ', ' '],
#  [' ', '>', '>', '>', '>', 'v'],
#  [' ', ' ', ' ', ' ', ' ', 'v'],
#  [' ', ' ', ' ', ' ', ' ', 'v'],
#  [' ', ' ', ' ', ' ', ' ', '*']]
#
# Where '>', '<', '^', and 'v' refer to right, left, 
# up, and down motions. Note that the 'v' should be 
# lowercase. '*' should mark the goal cell.
#
# You may assume that all test cases for this function
# will have a path from init to goal.
# ----------

grid = [[0, 0, 1, 0, 0, 0],
        [0, 0, 0, 0, 0, 0],
        [0, 0, 1, 0, 1, 0],
        [0, 0, 1, 0, 1, 0],
        [0, 0, 1, 0, 1, 0]]
init = [0, 0]
goal = [len(grid)-1, len(grid[0])-1]
cost = 1

delta = [[-1, 0 ], # go up
         [ 0, -1], # go left
         [ 1, 0 ], # go down
         [ 0, 1 ]] # go right

delta_name = ['^', '<', 'v', '>']

def search3(grid,init,goal,cost):
    # ----------------------------------------
    # modify code below
    # ----------------------------------------
    closed = [[-1 for row in range(len(grid[0]))] for col in range(len(grid))]
    closed[init[0]][init[1]] = 0
    expand = [[' ' for row in range(len(grid[0]))] for col in range(len(grid))]
    expand[goal[0]][goal[1]] = '*'
    x = init[0]
    y = init[1]
    g = 0

    open = [[g, x, y]]

    found = False  # flag that is set when search is complete
    resign = False # flag set if we can't find expand

    while not found and not resign:
        if len(open) == 0:
            resign = True
            return 'fail'
        else:
            open.sort()
            open.reverse()
            next = open.pop()
            x = next[1]
            y = next[2]
            g = next[0]
            
            if x == goal[0] and y == goal[1]:
                found = True
            else:
                for i in range(len(delta)):
                    x2 = x + delta[i][0]
                    y2 = y + delta[i][1]
                    if x2 >= 0 and x2 < len(grid) and y2 >=0 and y2 < len(grid[0]):
                        if closed[x2][y2] == -1 and grid[x2][y2] == 0:
                            g2 = g + cost
                            open.append([g2, x2, y2])
                            closed[x2][y2] = g2
                            
    delta_name2 = ['v', '>', '^', '<']
    x = goal[0]
    y = goal[1]
    g = closed[x][y]
    back_trace = [g, x, y]
    end = False
    while not end:
        prev_value = back_trace[0]-1
        x = back_trace[1]
        y = back_trace[2]
        for i in range(len(delta)):
            x2 = x + delta[i][0]
            y2 = y + delta[i][1]
            if x2 >= 0 and x2 < len(grid) and y2 >=0 and y2 < len(grid[0]):
                check_value = closed[x2][y2]
            
            if check_value == prev_value:
                back_trace = [prev_value, x2, y2]
                expand[x2][y2] = delta_name2[i]
                break
        if prev_value == 0:
            end = True
        
                
                       
    

    return expand  # make sure you return the shortest path

expand3 = search3(grid,init,goal,cost)
        

 
# -----------
# User Instructions:
#
# Modify the the search function so that it becomes
# an A* search algorithm as defined in the previous
# lectures.
#
# Your function should return the expanded grid
# which shows, for each element, the count when
# it was expanded or -1 if the element was never expanded.
# 
# If there is no path from init to goal,
# the function should return the string 'fail'
# ----------

grid = [[0, 1, 0, 0, 0, 0],
        [0, 1, 0, 1, 1, 0],
        [0, 1, 0, 0, 1, 0],
        [0, 1, 1, 0, 1, 0],
        [0, 0, 0, 0, 0, 0]]
heuristic = [[9, 8, 7, 6, 5, 4],
             [8, 7, 6, 5, 4, 3],
             [7, 6, 5, 4, 3, 2],
             [6, 5, 4, 3, 2, 1],
             [5, 4, 3, 2, 1, 0]]

init = [0, 0]
goal = [len(grid)-1, len(grid[0])-1]
cost = 1

delta = [[-1, 0 ], # go up
         [ 0, -1], # go left
         [ 1, 0 ], # go down
         [ 0, 1 ]] # go right

delta_name = ['^', '<', 'v', '>']

def search4(grid,init,goal,cost,heuristic):
    # ----------------------------------------
    # modify the code below
    # ----------------------------------------
    closed = [[0 for col in range(len(grid[0]))] for row in range(len(grid))]
    closed[init[0]][init[1]] = 1

    expand = [[-1 for col in range(len(grid[0]))] for row in range(len(grid))]
    action = [[-1 for col in range(len(grid[0]))] for row in range(len(grid))]

    x = init[0]
    y = init[1]
    g = 0
    f = g+ heuristic[x][y]

    open = [[f, g, x, y]]

    found = False  # flag that is set when search is complete
    resign = False # flag set if we can't find expand
    count = 0
    
    while not found and not resign:
        if len(open) == 0:
            resign = True
            return "Fail"
        else:
            open.sort()
            open.reverse()
            next = open.pop()
            x = next[2]
            y = next[3]
            g = next[1]
            f = next[0]
            expand[x][y] = count
            count += 1
            
            if x == goal[0] and y == goal[1]:
                found = True
            else:
                for i in range(len(delta)):
                    x2 = x + delta[i][0]
                    y2 = y + delta[i][1]
                    if x2 >= 0 and x2 < len(grid) and y2 >=0 and y2 < len(grid[0]):
                        if closed[x2][y2] == 0 and grid[x2][y2] == 0:
                            g2 = g + cost
                            f2 = g2 + heuristic[x2][y2]
                            open.append([f2, g2, x2, y2])
                            closed[x2][y2] = 1
    print expand
    print closed
    return expand
    
expand4 = search4(grid,init,goal,cost,heuristic)        
        

# ----------
# User Instructions:
# 
# Create a function compute_value which returns
# a grid of values. The value of a cell is the minimum
# number of moves required to get from the cell to the goal. 
#
# If a cell is a wall or it is impossible to reach the goal from a cell,
# assign that cell a value of 99.
# ----------

grid = [[0, 1, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 0]]
goal = [len(grid)-1, len(grid[0])-1]
cost = 1 # the cost associated with moving from a cell to an adjacent one

delta = [[-1, 0 ], # go up
         [ 0, -1], # go left
         [ 1, 0 ], # go down
         [ 0, 1 ]] # go right

delta_name = ['^', '<', 'v', '>']

def compute_value(grid,goal,cost):
    # ----------------------------------------
    # insert code below
    # ----------------------------------------
    
    x = goal[0]
    y = goal[1]
    g = 0
    open = [[g, x, y]]
    value = [[99 for col in range(len(grid[0]))] for row in range(len(grid))]
    closed = [[0 for col in range(len(grid[0]))] for row in range(len(grid))]
    closed[goal[0]][goal[1]] = 1

    finish  = False    
    while not finish:
        if len(open) == 0:
            finish = True
        
        else:
            open.sort()
            open.reverse()
            next = open.pop()
            x = next[1]
            y = next[2]
            g = next[0]
            value[x][y] = g
            for i in range(len(delta)):
                    x2 = x + delta[i][0]
                    y2 = y + delta[i][1]
                    if x2 >= 0 and x2 < len(grid) and y2 >=0 and y2 < len(grid[0]):
                        if closed[x2][y2] == 0 and grid[x2][y2] == 0:
                            g2 = g + cost
                            closed[x2][y2] = 1
                            open.append([g2, x2, y2])
                            
            
    # make sure your function returns a grid of values as 
    # demonstrated in the previous video.
    return value 
    

value2 = compute_value(grid,goal,cost)               


# ----------
#optimum_policy: from my own value method
# User Instructions:
# 
# Write a function optimum_policy that returns
# a grid which shows the optimum policy for robot
# motion. This means there should be an optimum
# direction associated with each navigable cell from
# which the goal can be reached.
# 
# Unnavigable cells as well as cells from which 
# the goal cannot be reached should have a string 
# containing a single space (' '), as shown in the 
# previous video. The goal cell should have '*'.
# ----------

grid = [[0, 1, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 0]]
init = [0, 0]
goal = [len(grid)-1, len(grid[0])-1]
cost = 1 # the cost associated with moving from a cell to an adjacent one

delta = [[-1, 0 ], # go up
         [ 0, -1], # go left
         [ 1, 0 ], # go down
         [ 0, 1 ]] # go right

delta_name = ['^', '<', 'v', '>']


def optimum_policy(grid,goal,cost):
    # ----------------------------------------
    # modify code below
    # ----------------------------------------
    delta_name2 = ['v', '>', '^', '<']
    x = goal[0]
    y = goal[1]
    g = 0
    open = [[g, x, y]]
    value = [[99 for col in range(len(grid[0]))] for row in range(len(grid))]
    closed = [[0 for col in range(len(grid[0]))] for row in range(len(grid))]
    policy = [[' ' for col in range(len(grid[0]))] for row in range(len(grid))]
    closed[goal[0]][goal[1]] = 1
    policy[goal[0]][goal[1]] = '*'

    finish  = False    
    while not finish:
        if len(open) == 0:
            finish = True
        
        else:
            open.sort()
            open.reverse()
            next = open.pop()
            x = next[1]
            y = next[2]
            g = next[0]
            value[x][y] = g
            for i in range(len(delta)):
                    x2 = x + delta[i][0]
                    y2 = y + delta[i][1]
                    if x2 >= 0 and x2 < len(grid) and y2 >=0 and y2 < len(grid[0]):
                        if closed[x2][y2] == 0 and grid[x2][y2] == 0:
                            g2 = g + cost
                            closed[x2][y2] = 1
                            policy[x2][y2] = delta_name2[i]
                            open.append([g2, x2, y2])
                            
            
    # make sure your function returns a grid of values as 
    # demonstrated in the previous video.
    return policy, closed 
    

policy, closed = optimum_policy(grid,goal,cost)   


# ----------
# User Instructions:
# 
# Write a function optimum_policy that returns
# a grid which shows the optimum policy for robot
# motion. This means there should be an optimum
# direction associated with each navigable cell from
# which the goal can be reached.
# 
# Unnavigable cells as well as cells from which 
# the goal cannot be reached should have a string 
# containing a single space (' '), as shown in the 
# previous video. The goal cell should have '*'.
# ----------

grid = [[0, 1, 0, 0, 0, 0],
        [0, 1, 1, 0, 1, 0],
        [0, 0, 0, 0, 1, 0],
        [0, 1, 1, 1, 1, 0],
        [0, 1, 0, 1, 1, 0]]
init = [0, 0]
goal = [len(grid)-1, len(grid[0])-1]
cost = 1 # the cost associated with moving from a cell to an adjacent one

delta = [[-1, 0 ], # go up
         [ 0, -1], # go left
         [ 1, 0 ], # go down
         [ 0, 1 ]] # go right

delta_name = ['^', '<', 'v', '>']

def optimum_policy2(grid,goal,cost):
    # ----------------------------------------
    # modify code below
    # ----------------------------------------
    value = [[99 for row in range(len(grid[0]))] for col in range(len(grid))]
    policy2 = [[' ' for col in range(len(grid[0]))] for row in range(len(grid))]
    change = True

    while change:
        change = False

        for x in range(len(grid)):
            for y in range(len(grid[0])):
                if goal[0] == x and goal[1] == y:
                    if value[x][y] > 0:
                        value[x][y] = 0
                        policy[x][y] = '*'

                        change = True

                elif grid[x][y] == 0:
                    for a in range(len(delta)):
                        x2 = x + delta[a][0]
                        y2 = y + delta[a][1]

                        if x2 >= 0 and x2 < len(grid) and y2 >= 0 and y2 < len(grid[0]) and grid[x2][y2] == 0:
                            v2 = value[x2][y2] + cost


                            if v2 < value[x][y]:
                                change = True
                                value[x][y] = v2
                                policy2[x][y] = delta_name[a]

    return policy2
    
policy2 = optimum_policy2(grid,goal,cost)


# ----------
# User Instructions:
# 
# Implement the function optimum_policy2D below.
#
# You are given a car in grid with initial state
# init. Your task is to compute and return the car's 
# optimal path to the position specified in goal; 
# the costs for each motion are as defined in cost.
#
# There are four motion directions: up, left, down, and right.
# Increasing the index in this array corresponds to making a
# a left turn, and decreasing the index corresponds to making a 
# right turn.

forward = [[-1,  0], # go up
           [ 0, -1], # go left
           [ 1,  0], # go down
           [ 0,  1]] # go right
forward_name = ['up', 'left', 'down', 'right']

# action has 3 values: right turn, no turn, left turn
action = [-1, 0, 1]
action_name = ['R', '#', 'L']

# EXAMPLE INPUTS:
# grid format:
#     0 = navigable space
#     1 = unnavigable space 
grid = [[1, 1, 1, 0, 0, 0],
        [1, 1, 1, 0, 1, 0],
        [0, 0, 0, 0, 0, 0],
        [1, 1, 1, 0, 1, 1],
        [1, 1, 1, 0, 1, 1]]

init = [4, 3, 0] # given in the form [row,col,direction]
                 # direction = 0: up
                 #             1: left
                 #             2: down
                 #             3: right
                
goal = [2, 0] # given in the form [row,col]

cost = [2, 1, 20] # cost has 3 values, corresponding to making 
                  # a right turn, no turn, and a left turn

# EXAMPLE OUTPUT:
# calling optimum_policy2D with the given parameters should return 
# [[' ', ' ', ' ', 'R', '#', 'R'],
#  [' ', ' ', ' ', '#', ' ', '#'],
#  ['*', '#', '#', '#', '#', 'R'],
#  [' ', ' ', ' ', '#', ' ', ' '],
#  [' ', ' ', ' ', '#', ' ', ' ']]
# ----------

# ----------------------------------------
# modify code below
# ----------------------------------------

def optimum_policy2D(grid,init,goal,cost):
    forward = [[-1,  0], # go up
           [ 0, -1], # go left
           [ 1,  0], # go down
           [ 0,  1]] # go right
    policy2D = [[' ' for col in range(len(grid[0]))] for row in range(len(grid))]
    policy2D[goal[0]][goal[1]] = '*'
    value = [[[9999 for arrow in range(len(forward)) ] for col in range(len(grid[0]))] for row in range(len(grid))]
    move = [[[' ' for arrow in range(len(forward)) ] for col in range(len(grid[0]))] for row in range(len(grid))]
    action = [-1, 0, 1]
    action_name = ['R', '#', 'L']


    finish = False

    while not finish:
 
        finish = True 
#    for trial_run in range(2009):     
        for x in range(len(grid)):
            for y in range(len(grid[0])):
                if x == goal[0] and y == goal[1]:
                    if value[x][y][1] >0:
                        value[x][y][:] = [0]*4
                        move[x][y][:] = ['*']*4

                        finish = False
                        print finish, x, y
                    
                
                elif grid[x][y] == 0:                    
                    for a in range(len(forward)):
                        
                        x2 = x + delta[a][0]
                        y2 = y + delta[a][1]
                        if x2 >= 0 and x2 < len(grid) and y2 >= 0 and y2 < len(grid[0]) and grid[x2][y2] == 0:

                            for b in range(len(forward)):
                                if delta[b] == delta[a]:
                                    if value[x2][y2][a]+cost[1] < value[x][y][b]:
                                    
                                        value[x][y][b]= value[x2][y2][a]+cost[1]
                                        move[x][y][b] = '#'
                                        
                                        finish = False
                                
                                elif delta[b][0] == 0:
                                    if delta[b][1]*delta[a][0] == 1:
                                        if value[x2][y2][a]+cost[0] < value[x][y][b]:
                                            value[x][y][b] = value[x2][y2][a]+cost[0]
                                            move[x][y][b] = 'R'
                                            finish = False
                                            
                                    elif delta[b][1]*delta[a][0] == -1:
                                        if value[x2][y2][a]+cost[2] < value[x][y][b]:
                                            value[x][y][b] = value[x2][y2][a]+cost[2]
                                            move[x][y][b] = 'L'                                           
                                            finish = False
                                    


                                elif delta[b][1] == 0:
                                    if delta[b][0]*delta[a][1] == 1:
                                        if value[x2][y2][a]+cost[2] < value[x][y][b]:
                                            value[x][y][b] = value[x2][y2][a]+cost[2]
                                            move[x][y][b] = 'L'                                            
                                            finish = False
                                    elif delta[b][0]*delta[a][1] == -1:
                                        if value[x2][y2][a]+cost[0] < value[x][y][b]:
                                            value[x][y][b] = value[x2][y2][a]+cost[0]
                                            move[x][y][b] = 'R'                                                   
                                            finish = False
    arrive = False
    while not arrive:
        x = init[0]
        y = init[1]
        direction = init[2] 
        print x,y, direction
        if value[x][y][direction] == 9999:
            print x,y, direction, policy2D
            arrive = True
            raise ValueError, "starting point cannot reach the final destination"
        
        else:
            policy2D[x][y] = move[x][y][direction]
            ind = action_name.index(policy2D[x][y])
            direction += action[ind]
            x+= forward[direction][0]
            y+= forward[direction][1]
            init = [x,y,direction]
            if policy2D[x][y] == '*':
                arrive = True
         
                                
                            
                                
                                

    return value, policy2D
    
value, policy2D = optimum_policy2D(grid,init,goal,cost)
