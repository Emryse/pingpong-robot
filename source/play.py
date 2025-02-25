import gym
from gym import envs
import gym_env.pingpong_env as pingpong
import warnings

#warnings.filterwarnings("ignore")

# 显示gym中所有envs
for env in envs.registry.all():
    print(env.id)

# 训练轮数
n_episodes = 100

env = gym.make('pingpong-v0')

for i in range(n_episodes):
    state = env.reset()
    while True:
        env.render()
        action = env.action_space.sample()
        observation, reward, done, info = env.step(action)
        print(f"""episode {i}: {observation}, {reward}, {done}, {info}""")
        if done:
            break
