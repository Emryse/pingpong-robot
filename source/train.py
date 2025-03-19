import gym
from gym import envs
import gym_env.pingpong_env as pingpong
import warnings
from agent.ppo_agent import Agent

#warnings.filterwarnings("ignore")

# 显示gym中所有envs
for env in envs.registry.all():
    print(env.id)

# 总训练轮数
n_episodes = 100
# 每次学习周期里，对N个历史轨迹分组时，每组的轨迹数量
batch_size = 5
# 每个学习周期里，使用该周期记录的历史轨迹，更新策略的次数（on-policy)
n_epochs = 4
# 策略网络的学习率，SGD随机梯度下降的参数变化比例
alpha = 0.0003
# 每10步更新一次网络
N = 10
# 总模拟步数
total_steps_n = 0
# 总学习次数
total_learn_n = 0

env = gym.make('pingpong-v0')

agent = Agent(n_actions=env.action_space.shape[0], batch_size=batch_size,
                alpha=alpha, n_epochs=n_epochs,
                input_dims=env.observation_space.shape)

for i in range(n_episodes):
    # 开始训练周期
    # 重置环境状态
    state = env.reset()
    score = 0
    done = False
    while not done:
        # 基于当前状态，预测下一步action、action的概率密度、期望回报值
        action, prob, val = agent.choose_action(state)
        # 基于预测的action, 计算下一步的状态、回报值
        state_, reward, done, info = env.step(action)
        # 渲染图形
        env.render()
        total_steps_n += 1

        # 记录轨迹轨迹
        agent.remember(state, action, prob, val, reward, done)
        # 每N次更新策略
        if total_steps_n % N == 0:
            agent.learn()
            total_learn_n += 1

        # 新状态作为当前状态
        state = state_
        print(f"""episode {i}: {state}, {reward}, {done}, {info}""")
        if done:
            print('\n')
