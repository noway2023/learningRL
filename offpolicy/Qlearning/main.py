import gym
from agent import QlearningAgent
from utils import *

# print(gym.envs.registry.keys())
env_name = "FrozenLake-v1"#"CliffWalking-v0"
env = gym.make(env_name)#, render_mode="human")


agent = QlearningAgent(obs_n=env.observation_space.n,
                   act_n=env.action_space.n,
                   learning_rate=0.1,
                   gamma=0.95,
                   egreed=0.1)
agent.load()

for i in range(10000):
    total_step, total_reward = run_a_epoch(env, agent)
    print(f"i {i} step {total_step}, reward {total_reward}")

agent.save()

env = gym.make(env_name, render_mode="human")
test_episode(env, agent)

env.close()
    