
from enum import Enum, IntEnum
import matplotlib.pyplot as plt
from q_learning import *
import numpy as np

from abstractmodel import *


class Cell(IntEnum):
    Free = 1  # Free cell
    Wall = 0  # Wall cell
# ALL possible moves 
class Action(IntEnum):
    Left = 0
    Right = 1
    Up = 2
    Down = 3
#All status of the game
class Status(Enum):
    Win = 0
    Lost = 1
    Play = 2
class Render(Enum):
    Inactive = 0
    Train = 1
    Move = 2

class Maze:

    actions = [Action.Left, Action.Right, Action.Up, Action.Down] 
    rew_exit = 10.0  # find the goal
    penalty_move = -0.04  # wanderind penalty
    penalty_visited = -0.25  # penalty for already visited cell
    penalty_impossible_move = -0.75  # penalty for trying to enter a Wall cell
    
    #Create a new maze game.
    def __init__(self, maze, start_cell=(1, 1), goal_cell=None):
  
        self.maze = maze
        self.__min_rew = -0.5 * self.maze.size  # down threshold
        n_rows, n_cols = self.maze.shape
        self.cells = [(i, j) for i in range(n_cols) for j in range(n_rows)]
        self.Free = [(i, j) for i in range(n_cols) for j in range(n_rows) if self.maze[j, i] == Cell.Free]
        self.__goal_cell = (n_cols - 1, n_rows - 1) if goal_cell is None else goal_cell
        self.Free.remove(self.__goal_cell)

        self.__render = Render.Inactive  
        self.__ax1 = None  
        self.__ax2 = None  
        self.reset(start_cell)

    # Reset the maze and place the agent at start_cell.
    def reset(self, start_cell=(1, 1)):
        
        self.__prev_cell = self.__curr_cell = start_cell
        self.__tot_rew = 0.0  # accumulated reward
        self.__already_visited = set() 

         # Graphycall canvas
        if self.__render in (Render.Train, Render.Move):
            n_rows, n_cols = self.maze.shape
            self.__ax1.clear()
            self.__ax1.set_xticks(np.arange(0.5, n_rows, step=1))
            self.__ax1.set_xticklabels([])
            self.__ax1.set_yticks(np.arange(0.5, n_cols, step=1))
            self.__ax1.set_yticklabels([])
            self.__ax1.grid(True)
            self.__ax1.plot(*self.__curr_cell, "rs", markersize=0.1) 
            self.__ax1.text(*self.__curr_cell, "Start", ha="center", va="center", color="white")
            self.__ax1.plot(*self.__goal_cell, "gs", markersize=0.1) 
            self.__ax1.text(*self.__goal_cell, "Exit", ha="center", va="center", color="white")
            self.__ax1.imshow(self.maze, cmap="binary")
            self.__ax1.get_figure().canvas.draw()
            self.__ax1.get_figure().canvas.flush_events()
        ret = np.array([[*self.__curr_cell]])
        return ret
        
    # Next step of the agent and create the new state, reward and game status.
    def step(self, action):
    
        #Possible Actions
        if self.__curr_cell is None:
            col, row = self.__curr_cell
        else:
            col, row = self.__curr_cell

        pos_action = Maze.actions.copy() 
        n_rows, n_cols = self.maze.shape
        if row == 0 or (row > 0 and self.maze[row - 1, col] == Cell.Wall):
            pos_action.remove(Action.Up)
        if row == n_rows - 1 or (row < n_rows - 1 and self.maze[row + 1, col] == Cell.Wall):
            pos_action.remove(Action.Down)

        if col == 0 or (col > 0 and self.maze[row, col - 1] == Cell.Wall):
            pos_action.remove(Action.Left)
        if col == n_cols - 1 or (col < n_cols - 1 and self.maze[row, col + 1] == Cell.Wall):
            pos_action.remove(Action.Right)


        if not pos_action:
            rew = self.__min_rew - 1  # end of the game, thera are no any move
        elif action in pos_action:
            col, row = self.__curr_cell
            if action == Action.Left:
                col -= 1
            elif action == Action.Up:
                row -= 1
            if action == Action.Right:
                col += 1
            elif action == Action.Down:
                row += 1

            self.__prev_cell = self.__curr_cell
            self.__curr_cell = (col, row)
            if self.__render != Render.Inactive:
                self.__ax1.plot(*zip(*[self.__prev_cell, self.__curr_cell]), "bo-", markersize=0.1)  # path
                self.__ax1.plot(*self.__curr_cell, "ro", markersize=0.1)  
                self.__ax1.get_figure().canvas.draw()
                self.__ax1.get_figure().canvas.flush_events()

            if self.__curr_cell == self.__goal_cell:
                rew = Maze.rew_exit  # reward of reaching the goal cell
            elif self.__curr_cell in self.__already_visited:
                rew = Maze.penalty_visited  # penalty for visited cell
            else:
                rew = Maze.penalty_move  # wandering penalty
            self.__already_visited.add(self.__curr_cell)
        else:
            rew = Maze.penalty_impossible_move  # wall penalty

        self.__tot_rew += rew
        if self.__tot_rew < self.__min_rew:  # reached the threshold
            status = Status.Lost
        elif self.__curr_cell == self.__goal_cell:
            status = Status.Win
        else:
            status = Status.Play
        state = np.array([[*self.__curr_cell]])
        return state, rew, status

    
    #  GRAPHICAL ANNIMATION FUNCTIONS  (found at github)
    
    def play(self, model, start_cell=(1, 1)):
        self.reset(start_cell)
        state = np.array([[*self.__curr_cell]])
        actions = []
        while True:
            action = model.predict(state=state)
            actions.append(action)
            state, rew, status = self.step(action)
            # if action == 0 :
            #   print ('left')
            # if action == 1 :
            #   print ('Right')
            # if action == 2 :
            #   print ('Up')
            # if action == 3 :
            #   print ('Down')
            if status in (Status.Win, Status.Lost):
                return status, actions

    def render(self, cont=Render.Inactive):
        self.__render = cont
        if self.__render in (Render.Move, Render.Train):
            if self.__ax1 is None:
                fig, self.__ax1 = plt.subplots(1, 1, tight_layout=True)
                fig.canvas.set_window_title("Maze")

        plt.show(block=False)

    def render_q_value(self, model):
        def clip(n):
            return max(min(1, n), 0)

        if self.__render == Render.Train:
            n_rows, n_cols = self.maze.shape

            self.__ax2.clear()
            self.__ax2.set_xticks(np.arange(0.5, n_rows, step=1))
            self.__ax2.set_xticklabels([])
            self.__ax2.set_yticks(np.arange(0.5, n_cols, step=1))
            self.__ax2.set_yticklabels([])
            self.__ax2.grid(True)
            self.__ax2.plot(*self.__goal_cell, "gs", markersize=0.1)  # exit is a big green sq_valueuare
            self.__ax2.text(*self.__goal_cell, "Exit", ha="center", va="center", color="white")

            for cell in self.Free:
                q_value = model.q_value(cell) if model is not None else [0, 0, 0, 0]
                a = np.nonzero(q_value == np.max(q_value))[0]
                for action in a:
                    dx = 0
                    dy = 0
                    if action == Action.Left:
                        dx = -0.2
                    if action == Action.Right:
                        dx = +0.2
                    if action == Action.Up:
                        dy = -0.2
                    if action == Action.Down:
                        dy = 0.2
                    color = clip((q_value[action] - -1)/(1 - -1))
                    self.__ax2.arrow(*cell, dx, dy, color=(1 - color, color, 0), head_width=0.2, head_length=0.1)
            self.__ax2.imshow(self.maze, cmap="binary")
            self.__ax2.get_figure().canvas.draw()


