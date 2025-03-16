import sys
import math
from typing import Optional
import gym
from gym import spaces, logger
from gym.utils import seeding
from gym.utils import seeding, EzPickle
import numpy as np
import mujoco

import pygame
from networkx.algorithms.bipartite.basic import color, density
from pygame import gfxdraw

from gym_env.b2drawer import B2Drawer
from gym_env.utils import world_to_screen, screen_to_world, FPS, PPM, TABLE_W, TABLE_H, COURT_W, COURT_H, SCREEN_W, SCREEN_H

import Box2D
from Box2D  import b2LoopShape, b2PolygonShape, b2EdgeShape, b2Vec2, b2ContactListener, b2FixtureDef, b2PolygonShape
from gym_env.pygame_spirite import Ball, Bat

# 场地范围
COURT_POLY = [(0, 0), (COURT_W, 0), (COURT_W, COURT_H), (0, COURT_H)]

class FrictionDetector(b2ContactListener):
    def __init__(self, env):
        b2ContactListener.__init__(self)
        self.env = env

    def BeginContact(self, contact):
        fixtureA = contact.fixtureA
        fixtureB = contact.fixtureB
        # 获取物体和 fixture 的标识符
        bodyA = fixtureA.body
        bodyB = fixtureB.body
        userDataArray = (bodyA.userData, bodyB.userData)

        # 乒乓球碰撞到场地外墙，超过最大次数则结束
        if 'ball' in userDataArray and 'court' in userDataArray:
            self.env.ball_contact_court_n += 1
            if self.env.ball_contact_court_n > self.env.ball_contact_court_max:
                self.env.is_open = False

class PingPongEnv(gym.Env):
    """
    Env Constants:
    Bat mass
    Air resistance

    Action space:
    | Num | Action                    | Range       |
    | 0   | The force pushing the bat | [-100, 100] N, Positive value push right |
    | 1   | Bat angle with horizon    | [-90°, 90°]  |

    Observation space:
    | Num |
    | 0   | Bat Horizon Position | [-1.0, 1.0] m |
    | 1   | Ball Position X      |
    | 2   | Ball Position Y      |

    """
    # 球拍上施加的最大力 N
    bat_force_max = 5
    # 球拍最大转动角度
    bat_angle_max = 90

    # 求最多碰撞场地外墙的次数，超过则结束。碰撞越少则奖励值越高
    ball_contact_court_max = 5

    # 球和球拍的坐标原点移动到 球桌下边缘中点
    translation = (int(COURT_W / 2), (COURT_H - TABLE_H) / 2)

    metadata = {
        "render_modes": ["human", "rgb_array", "state_pixels"],
        "render_fps": FPS,
    }

    def __init__(self):
        pygame.init()
        self.contactListener = FrictionDetector(self)
        self.world = Box2D.b2World((0.0, -9.81), contactListener=self.contactListener)
        self.screen = None
        self.clock = None
        self.is_open = True
        self.total_time = 0.0
        # 球碰撞场地外墙次数，最多ball_contact_court_max
        self.ball_contact_court_n = 0

        # 定义场地， v0版本为垂直抛球接球
        self.court = self.world.CreateBody(
            position=(0, 0),
            shapes=[b2EdgeShape(vertices=COURT_POLY[0:2]),
                    b2EdgeShape(vertices=COURT_POLY[1:3]),
                    b2EdgeShape(vertices=COURT_POLY[2:]),
                    b2EdgeShape(vertices=[COURT_POLY[-1], COURT_POLY[-0]])],
            #shapes=b2LoopShape(vertices=WALL_POLY),
            shapeFixture=b2FixtureDef(
                density=10.0,
                restitution=0.9,  # 反弹系数（0~1）
                friction=0.1  # 降低摩擦增强反弹效果
            ),
            userData = "court",
        )

        # 定义球桌，v0版本为垂直抛球接球，不需要球桌
        self.table = self.world.CreateStaticBody(
            position=(COURT_W/ 2, COURT_H/ 2),
            shapes=b2PolygonShape(box=(TABLE_W/2, TABLE_H/2)),
            userData="table",
        )
        # 设置球桌为感应器类型，俯视外边界不发生物理碰撞
        for fixture in self.table.fixtures:
            fixture.sensor = True

        # 定义乒乓球
        self.ball = Ball(self.world, init_x=COURT_W/2, init_y=COURT_H)
        # 定义球拍
        self.bat = Bat(self.world, init_x=COURT_W/2, init_y=(COURT_H - TABLE_H)/2)

        # 球拍动作参数空间
        act_space_high = np.array(
            [
                self.bat_force_max,
                self.bat_angle_max,
            ],
            dtype=np.float32,
        )
        self.action_space = spaces.Box(-act_space_high, act_space_high, dtype=np.float32)

        # 环境状态参数空间
        obs_space_high = np.array(
            [
                COURT_W / 2,
                COURT_W / 2,
                COURT_H / 2,
            ],
            dtype=np.float32,
        )
        self.observation_space = spaces.Box(-obs_space_high, obs_space_high, dtype=np.float32)

    def reset(self, *,
        seed: Optional[int] = None,
        return_info: bool = False,
        options: Optional[dict] = None,
    ):
        super().reset()
        self.is_open = True
        # 总运行时间
        self.total_time = 0.0
        # 球碰撞场地外墙次数
        self.ball_contact_court_n = 0

        # 球拍初始位置x
        self.bat.init_x = COURT_W / 2
        self.bat.angle = 0.0
        self.bat.need_reset = True

        # 球初始位置, 自由落体
        self.ball.init_x = TABLE_W * np.random.rand()
        self.ball.init_y = COURT_H
        self.ball.need_reset = True

        return self.bat.init_x, self.ball.init_x, self.ball.init_y

    def step(self, action):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.is_open = False
                self.close()

            if event.type == pygame.MOUSEMOTION:  # 鼠标控制球拍
                world_pos = screen_to_world(event.pos)
                self.bat.position = b2Vec2(world_pos)  # Y轴固定高度

        if action is not None:
            # 从Agent返回的可能为pytorch.Tensor类型
            if type(action) != np.ndarray:
                action = action.numpy()

            # 将agent动作值按比例(0,1)映射到动作值空间(low, high)
            high = self.action_space.high
            low = self.action_space.low
            action = low + (high - low) * action
            self.bat.act(action)

        self.ball.step()
        self.bat.step()
        self.world.Step(1.0 / FPS, 6 * 30, 2 * 30)
        self.total_time += 1.0 / FPS

        #self.state = self.render("state_pixels")
        observation = (self.bat.body.position.x, self.ball.body.position.x, self.ball.body.position.y)

        # 计算奖励值
        reward = 1.0
        # 奖励值 反比于求碰撞场地外墙次数 ball_contact_court_n, 可以尝试结合球拍击中球次数、运行时间
        if self.is_open:
            reward = 0.0
        else:
            reward = 1 - self.ball_contact_court_n / self.ball_contact_court_max

        return observation, reward, not self.is_open, {}

    def seed(self, seed=None):
        super().seed(seed)

    def render(self, mode='human'):
        assert mode in ["human", "state_pixels", "rgb_array"]
        if self.screen is None and mode == "human":
            pygame.display.init()
            pygame.display.set_caption("Pygame Pingpong")
            self.screen = pygame.display.set_mode((SCREEN_W, SCREEN_H))
        if self.clock is None:
            self.clock = pygame.time.Clock()

        if "total_time" not in self.__dict__:
            return

        self.surf = pygame.Surface((SCREEN_W, SCREEN_H))

        # 绘制场地
        #B2Drawer.draw_body(self.court, self.surf)
        for i in range(4):
            pygame.draw.polygon(self.surf, color=(100, 100, 100), points=world_to_screen(self.court.fixtures[i].shape.vertices), width=20)
        # 绘制球桌，v0版本为垂直抛球接球，不需要球桌
        B2Drawer.draw_body(self.table, self.surf)

        self.ball.draw(self.surf)
        self.bat.draw(self.surf)
        #self.surf = pygame.transform.flip(self.surf, False, True)

        font = pygame.font.Font(pygame.font.get_default_font(), 26)
        text = font.render("pingpong-v0", True, (255, 255, 255), (0, 0, 0))
        text_rect = text.get_rect()
        text_rect.bottomleft = (15, text_rect.height + 15)
        self.surf.blit(text, text_rect)

        if mode == "human":
            pygame.event.pump()
            self.clock.tick(self.metadata["render_fps"])
            self.screen.fill(0)
            self.screen.blit(self.surf, (0, 0))
            pygame.display.flip()

    def close(self):
        super().close()
        pygame.quit()
        if self.screen is not None:
            pygame.display.quit()
            self.is_open = False

__all__ = ['PingPongEnv']
