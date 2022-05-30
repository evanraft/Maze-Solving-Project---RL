import logging
from enum import Enum, auto
import tensorflow as tf

import matplotlib.pyplot as plt
import numpy as np
from q_learning import Q_Learning
from read_maze import load_maze, get_local_maze_information
from maze import Maze, Render

logging.basicConfig(format="%(levelname)-8s: %(asctime)s: %(message)s",
                    datefmt="%Y-%m-%d %H:%M:%S",
                    level=logging.INFO)  # Only show messages *equal to or above* this level

# maze = np.array([
#     [0, 1, 0, 0, 0, 0, 0, 0],
#     [0, 0, 0, 1, 0, 1, 0, 0],
#     [0, 0, 0, 1, 1, 0, 1, 0],
#     [0, 1, 0, 1, 0, 0, 0, 0],
#     [1, 0, 0, 1, 0, 1, 0, 0],
#     [0, 0, 0, 1, 0, 1, 1, 1],
#     [0, 1, 1, 0, 0, 0, 0, 0],
#     [0, 0, 0, 0, 0, 1, 0, 0]
# ])  # 0 = free, 1 = occupied
#
# maze = np.array([
# [0, 1, 1, 0, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1],
# [0, 0, 1, 0, 0, 1, 0, 1, 1, 0, 1, 1, 0, 0, 1],
# [1, 0, 1, 1, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1],
# [1, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 0, 1],
# [1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 1, 1, 0, 1],
# [1, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 0, 1],
# [1, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1],
# [1, 1, 0, 0, 0, 0, 0, 1, 1, 0, 1, 0, 1, 1, 1],
# [0, 1, 0, 1, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
# [1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 0, 1, 1],
# [1, 0, 0, 1, 0, 0, 0, 0, 0, 1, 1, 1, 0, 1, 0],
# [1, 1, 0, 1, 1, 1, 1, 0, 1, 0, 0, 0, 0, 1, 0],
# [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
# [1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1],
# [1, 0, 1, 1, 1, 0, 1, 0, 0, 1, 1, 1, 0, 0, 0],
# ])

print ('Loading the maze')
load_maze()

# Create a 2D maze for solving the static maze. Because calling the get_local_maze_information every time
# requires much time. Thus it is more efficient to create the maze for solving the static maze.
maze=np.ones((201,201),int)
for i in range(0,201):
    for j in range(0,201):
        maze[i][j]=-5

for i in range(1,201,3):
    for j in range(1,201,3):
        inf=get_local_maze_information(i,j)
        inf1=get_local_maze_information(i,j+1)
        maze[i-1][j-1]=inf[0][0][0]
        maze[i-1][j]=inf[0][1][0]
        maze[i-1][j+1]=inf[0][2][0]
        maze[i][j-1]=inf[1][0][0]
        maze[i][j]=inf1[1][0][0]
        maze[i][j+1]=inf[1][2][0]
        maze[i+1][j-1]=inf[2][0][0]
        maze[i+1][j]=inf[2][1][0]
        maze[i+1][j+1]=inf[2][2][0]


game = Maze(maze, start_cell=(1, 1), goal_cell=(199,199))
#print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

# Training the model
model = Q_Learning(game, name="Q_Learning")
h, w = model.train(disc=0.90, exp_rate=0.2, episodes=500)

# list thet contains the moves from the beginning (1,1) to the goal (199,199)
moves = []
_, actions = game.play(model, start_cell=(1, 1))
for i in actions:
    if i == 0:
        moves.append('Left')
    if i == 1:
        moves.append('Right')
    if i == 2:
        moves.append('Up')
    if i == 3:
        moves.append('Down')
print(moves)

# Graphycal annimations (take approximately one hour to finish the steps to the goal)
game.render(Render.Move)
game.play(model, start_cell=(1, 1))

plt.show()
