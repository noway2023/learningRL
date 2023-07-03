import numpy as np
import time

class QlearningAgent(object):
    def __init__(self, obs_n, act_n, learning_rate=0.01, gamma=0.9, egreed=0.1):
        self.lr = learning_rate
        self.gamma = gamma
        self.epsilon = egreed
        self.act_n = act_n
        self.obs_n = obs_n
        self.Q = np.zeros((obs_n,act_n))
        
    def testsample(self, obs):
        action = self.predict(obs)
        
        return action
    
    def sample(self, obs):
        if (np.random.uniform(0,1)< (1 - self.epsilon)):
            action = self.predict(obs)
        else:
            action = np.random.choice(self.act_n)
        return action
    
    def predict(self, obs):
        Q_list = self.Q[obs, :]
        maxQ = np.max(Q_list)
        action_list = np.where(Q_list == maxQ)[0]
        action = np.random.choice(action_list)
        return action        
    
    def learn(self, obs, action, reward, new_obs, terminate, truncted):

        predict_Q = self.Q[obs, action]
        if terminate or truncted:
            target_Q = reward
        else:
            target_Q = reward + self.gamma* np.max(self.Q[new_obs, :])
        self.Q[obs, action] += self.lr * (target_Q - predict_Q)
        
    def save(self):
        file = 'q_table.npy'
        np.save(file, self.Q, allow_pickle=True)
        print(file," saved!")
        
    def load(self, file='q_table.npy'):
        self.Q = np.load(file, allow_pickle=True)
        print(file, "load!")

        