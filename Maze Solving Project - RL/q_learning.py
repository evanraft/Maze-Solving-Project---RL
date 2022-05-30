import logging
import random
from maze import Status
import numpy as np
from abstractmodel import AbstractModel


class Q_Learning(AbstractModel):

    def __init__(self, game, **kwargs):
        super().__init__(game, **kwargs)
        self.Q = dict()  # the Q-table as dictionary. Keys are the (state+action)

    def q_value(self, state):
        #Get q_value values for all actions for a state
        if type(state) == np.ndarray:
            state = tuple(state.flatten())
        ret = np.array([self.Q.get((state, action), 0.0) for action in self.environment.actions])
        return ret

    def predict(self, state):
        # select the action with the max value from the Q-table. returns the next actiom
        q_value = self.q_value(state)
        actions = np.nonzero(q_value == np.max(q_value))[0]  # get index
        ret = random.choice(actions)
        return ret


    # Training function. Using Bellman function to find the next move. 
    # lr -> learning rate
    # disc -> the discount factor
    # exp_rate -> the explotration rate
    
    def train(self, disc, exp_rate, episodes):
        
        award_hist = []
        win_hist = []
        exp_dec = 0.995  # % reduction of exp_rate
        lr = 0.10
        total_award = 0
        episode=1
        flag=True
        while episode < episodes+1:
            # always start from the start_cell = (1,1)
            start_cell = (1, 1)
            #reset the state
            state = self.environment.reset(start_cell)
            state = tuple(state.flatten())
            while flag==True:
                # next action epsilon greedy (off-policy method)
                if exp_rate > np.random.random():
                    action=random.choice(self.environment.actions)
                else:
                    q_value = self.q_value(state)
                    actions = np.nonzero(q_value==np.max(q_value))[0] 
                    action = random.choice(actions)
                next_state,rew,status = self.environment.step(action)
                next_state = tuple(next_state.flatten())
                total_award = total_award + rew

                if (state,action) not in self.Q.keys(): 
                    self.Q[(state, action)] = 0.0

                max_q = max([self.Q.get((next_state, a), 0.0) for a in self.environment.actions])
                # Bellman function
                self.Q[(state, action)] += lr * (rew + disc * max_q - self.Q[(state, action)])
                # stop if find the goal or reach the threshold
                if status in (Status.Win, Status.Lost):  
                    break
                    flag==False
                state = next_state
                self.environment.render_q_value(self)
            award_hist.append(total_award)
            logging.info("episode: {:d}/{:d} | result: {:4s} | epsilon: {:.5f}".format(episode, episodes, status.name, exp_rate))
            exp_rate=exp_rate * exp_dec 
            episode += 1
        return award_hist, win_hist


