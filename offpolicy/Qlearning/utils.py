import numpy as np


def run_a_epoch(env, agent):
    total_step = 0
    total_reward = 0
    
    obs = env.reset()[0]

    while True:
        action = agent.sample(obs)
        new_obs, reward, terminate, truncted, info= env.step(action)
        
        agent.learn(obs, action, reward, new_obs, terminate, truncted)
        obs = new_obs
        
        total_step += 1
        total_reward += reward
        
        if terminate or truncted:
            break
        
    return total_step, total_reward

def test_episode(env, agent):
    total_step = 0
    total_reward = 0
    
    obs = env.reset()[0]
    print(agent.Q)
    for i in range(1000):
        action = agent.testsample(obs)
        obs, reward, terminate, truncted, info= env.step(action)
        
        total_reward += reward
        total_step += 1
        
        if terminate or truncted:
            print(f"step: {total_step},   test_reward {total_reward}")
            break
