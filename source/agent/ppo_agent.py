import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.categorical import Categorical
from torch.distributions import Beta

class PPOMemory:
    """
    经验池
    """
    def __init__(self, batch_size):
        self.states = []
        self.probs = []
        self.vals = []
        self.actions = []
        self.rewards = []
        self.dones = []

        self.batch_size = batch_size

    def generate_batches(self):
        n_states = len(self.states)
        batch_start = np.arange(0, n_states, self.batch_size)
        indices = np.arange(n_states, dtype=np.int64)
        np.random.shuffle(indices)
        batches = [indices[i:i + self.batch_size] for i in batch_start]

        return np.array(self.states), \
               np.array(self.actions), \
               np.array(self.probs), \
               np.array(self.vals), \
               np.array(self.rewards), \
               np.array(self.dones), \
               batches

    def store_memory(self, state, action, probs, vals, reward, done):
        self.states.append(state)
        self.actions.append(action)
        self.probs.append(probs)
        self.vals.append(vals)
        self.rewards.append(reward)
        self.dones.append(done)

    def clear_memory(self):
        self.states = []
        self.probs = []
        self.actions = []
        self.rewards = []
        self.dones = []
        self.vals = []


class ActorNetwork(nn.Module):
    """
    构建策略网络--actor
    """
    def __init__(self, n_actions, input_dims, alpha,
                 fc1_dims=256, fc2_dims=256, chkpt_dir='tmp/ppo'):
        super(ActorNetwork, self).__init__()

        self.checkpoint_file = os.path.join(chkpt_dir, 'actor_torch_ppo')
        # 策略网络，生成action的概率分布参数
        # 可以选择一种概率分布模型，比如此处选择 Beta分布 参数 alpha、beta 需大于0。也可以选择高斯分布 mu、sigma(std标准差)
        self.actor = nn.Sequential(
            nn.Linear(*input_dims, fc1_dims),
            nn.ReLU(),
            nn.Linear(fc1_dims, fc2_dims),
            nn.ReLU(),
            # 每个action输出 alpha、beta两个参数
            nn.Linear(fc2_dims, 2 * n_actions),
            # Relu限制值域[0, oo)，线性高效
            nn.ReLU()
        )

        self.optimizer = optim.Adam(self.parameters(), lr=alpha)
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state):
        """
        返回动作的取值 (-1, 1)范围内
        :param state:
        :return:
        """
        dist = self.actor(state)
        # 避免0
        dist = dist + 1e-5
        #dist = torch.exp(dist)

        return dist

    def save_checkpoint(self):
        """
        保存模型
        :return:
        """
        torch.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        """
        加载模型
        :return:
        """
        self.load_state_dict(torch.load(self.checkpoint_file))


class CriticNetwork(nn.Module):
    """
    构建价值网络--critic
    """
    def __init__(self, input_dims, alpha, fc1_dims=256, fc2_dims=256,
                 chkpt_dir='tmp/ppo'):
        super(CriticNetwork, self).__init__()

        self.checkpoint_file = os.path.join(chkpt_dir, 'critic_torch_ppo')
        self.critic = nn.Sequential(
            nn.Linear(*input_dims, fc1_dims),
            nn.ReLU(),
            nn.Linear(fc1_dims, fc2_dims),
            nn.ReLU(),
            nn.Linear(fc2_dims, 1)
        )

        self.optimizer = optim.Adam(self.parameters(), lr=alpha)
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state):
        value = self.critic(state)

        return value

    def save_checkpoint(self):
        """
        保存模型
        :return:
        """
        torch.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        """
        加载模型
        :return:
        """
        self.load_state_dict(torch.load(self.checkpoint_file))


class Agent:
    def __init__(self, n_actions, input_dims, gamma=0.99, alpha=0.0003, gae_lambda=0.95,
                 policy_clip=0.2, batch_size=64, n_epochs=10):
        self.gamma = gamma
        self.policy_clip = policy_clip
        self.n_epochs = n_epochs
        self.gae_lambda = gae_lambda
        # 实例化策略网络
        self.actor = ActorNetwork(n_actions, input_dims, alpha)
        # 实例化价值网络
        self.critic = CriticNetwork(input_dims, alpha)
        # 实例化经验池
        self.memory = PPOMemory(batch_size)

    def remember(self, state, action, probs, vals, reward, done):
        """
        记录轨迹
        :param state:
        :param action:
        :param probs:
        :param vals:
        :param reward:
        :param done:
        :return:
        """
        self.memory.store_memory(state, action, probs, vals, reward, done)

    def save_models(self):
        print('... saving models ...')
        self.actor.save_checkpoint()
        self.critic.save_checkpoint()

    def load_models(self):
        print('... loading models ...')
        self.actor.load_checkpoint()
        self.critic.load_checkpoint()

    def choose_action(self, observation):
        """
        选择动作
        :param observation:
        :return:
        """
        # 维度变换 [n_state]-->tensor[1,n_states]
        state = torch.tensor([observation], dtype=torch.float).to(self.actor.device)
        # 策略网络生成动作值概率函数参数，Beta分布 alpha、beta浓度
        dist = self.actor(state).detach()
        # 各个action参数的概率函数参数
        dist = dist.reshape(int(dist.size(1) / 2), 2)
        # Beta分布 [0,1]（适用于有界动作），也可以用高斯（正态）分布，但是需要clip超出值域的值
        dist = Beta(dist[:, 0], dist[:, 1])
        # 从概率分布中采样，让输出值在策略网络输出附近出现随机分布
        action = dist.sample()

        # 预测，当前状态的state_value  [b,1]
        # 为什么用评价网络生成期望回报值？
        value = self.critic(state).detach()

        # 输出值的概率密度的对数log(e,p)（log让概率密度在更广的(0, oo)域分布）
        probs = torch.squeeze(dist.log_prob(action))
        # 求解action中多维值的联合概率密度（条件概率求积）
        # 将action中多维值 的概率密度 求积，得到action的多值联合概率密度，在求log，等效于对概率密度的log值求和
        probs = torch.sum(probs).numpy()

        action = torch.squeeze(action).numpy()
        value = torch.squeeze(value).numpy()

        # 此处不转换输出值到外部值域，因为后面还需要用此值计算新策略的期望
        # 将action服从Beta分布的值 从 [0,1] 映射到 [-1,1]
        # 转换为numpy数组，不强制传递给env torch.Tensor类型
        # action = (action * 2.0 - 1.0).numpy()

        return action, probs, value

    def learn(self):
        # 每次学习需要更新n_epochs次参数，学习完后清空轨迹
        for _ in range(self.n_epochs):
            # 提取数据集，每个epoch从历史轨迹中，shuffle打乱，随机分成不重复的组
            state_arr, action_arr, old_prob_arr, vals_arr, \
            reward_arr, dones_arr, batches = \
                self.memory.generate_batches()

            # 计算优势函数，为每个轨迹计算优势函数
            values = vals_arr
            advantage = np.zeros(len(reward_arr), dtype=np.float32)

            for t in range(len(reward_arr) - 1): # 逆序时序差分值 axis=1轴上倒着取 [], [], []
                discount = 1
                a_t = 0
                for k in range(t, len(reward_arr) - 1):
                    a_t += discount * (reward_arr[k] + self.gamma * values[k + 1] * \
                                       (1 - int(dones_arr[k])) - values[k])
                    discount *= self.gamma * self.gae_lambda
                advantage[t] = a_t
            advantage = torch.tensor(advantage).to(self.actor.device)

            # 估计状态的值函数的数组
            values = torch.tensor(values).to(self.actor.device)

            for batch in batches:
                # 获取数据
                states = torch.tensor(state_arr[batch], dtype=torch.float).to(self.actor.device)
                old_probs = torch.tensor(old_prob_arr[batch]).to(self.actor.device)
                actions = torch.tensor(action_arr[batch]).to(self.actor.device)

                # 用当前网络进行预测
                dist = self.actor(states)
                # 各个action参数的概率函数参数
                dist = dist.reshape(dist.size(0) * int(dist.size(1) / 2), 2)
                # Beta分布，与choose_action保持一致
                dist = Beta(dist[:, 0], dist[:, 1])

                critic_value = self.critic(states)
                critic_value = torch.squeeze(critic_value)

                # 每一轮更新一次策略网络预测的状态
                # 新策略下原有actions的概率密度log
                new_probs = dist.log_prob(actions.flatten())
                # reshape为多个actions x 多值 形状的概率密度log
                new_probs = new_probs.reshape(batch.size, 2)
                # 求解action中多维值的联合概率密度（条件概率求积）
                # 将action中多维值 的概率密度 求积，得到action的多值联合概率密度，在求log，等效于对概率密度的log值求和
                new_probs = torch.sum(new_probs, dim=1)

                # 新旧策略之间的比例
                prob_ratio = new_probs.exp() / old_probs.exp()
                # 等效于 概率密度log 求差 后，e指数
                # prob_ratio = (new_probs - old_probs).exp()

                ### 根据新旧策略比例的阈值，对新旧策略比例进行裁剪clip
                # 近端策略优化裁剪目标函数公式的左侧项
                weighted_probs = advantage[batch] * prob_ratio
                # 公式的右侧项，ratio小于1-eps就输出1-eps，大于1+eps就输出1+eps
                weighted_clipped_probs = torch.clamp(prob_ratio, 1 - self.policy_clip,
                                                     1 + self.policy_clip) * advantage[batch]
                ### 计算损失值
                actor_loss = -torch.min(weighted_probs, weighted_clipped_probs).mean()
                returns = advantage[batch] + values[batch]
                critic_loss = (returns - critic_value) ** 2
                critic_loss = critic_loss.mean()

                total_loss = actor_loss + 0.5 * critic_loss

                ### 梯度下降，优化网络
                # 清除策略网络 和 评价网络的 梯度状态，反向传播
                self.actor.optimizer.zero_grad()
                self.critic.optimizer.zero_grad()
                total_loss.backward()
                # 优化策略和评价网络参数
                self.actor.optimizer.step()
                self.critic.optimizer.step()

        # 每个学习周期结束，清除历史轨迹
        self.memory.clear_memory()