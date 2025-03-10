import gym
from gym import envs
import gym_env.pingpong_env as pingpong
import warnings
from agent.ppo_agent import Agent

#warnings.filterwarnings("ignore")

# 显示gym中所有envs
for env in envs.registry.all():
    print(env.id)

# 训练轮数
n_episodes = 100
batch_size = 5
# 每次更新的次数
n_epochs = 4
# 学习率
alpha = 0.0003

env = gym.make('pingpong-v0')

agent = Agent(n_actions=env.action_space.shape[0], batch_size=batch_size,
                alpha=alpha, n_epochs=n_epochs,
                input_dims=env.observation_space.shape)

for i in range(n_episodes):
    state = env.reset()
    score = 0
    done = False
    while not done:
        #action = env.action_space.sample()
        action, prob, val = agent.choose_action(state)
        state, reward, done, info = env.step(action)
        env.render()

        # 存储轨迹
        agent.remember(state, action, prob, val, reward, done)
        # 更新网络
        agent.learn()
        #print(f"""episode {i}: {observation}, {reward}, {done}, {info}""")
