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
from gym_env.utils import world_to_screen, screen_to_world, PPM, TABLE_W, TABLE_H, COURT_W, COURT_H, SCREEN_W, SCREEN_H

import Box2D
from Box2D  import b2LoopShape, b2PolygonShape, b2EdgeShape, b2Vec2, b2ContactListener, b2FixtureDef, b2PolygonShape
from gym_env.pygame_spirite import Ball, Bat

# 场地范围
WALL_POLY = [(0, 0), (COURT_W, 0), (COURT_W, COURT_H), (0, COURT_H)]

class FrictionDetector(b2ContactListener):
    def __init__(self, env):
        b2ContactListener.__init__(self)
        self.env = env

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
    FPS = 50

    # 球拍上施加的最大力 N/m
    bat_force_max = 100
    # 球拍最大转动角度
    bat_angle_max = 90

    # 球和球拍的坐标原点移动到 球桌下边缘中点
    translation = (int(COURT_W / 2), (COURT_H - TABLE_H) / 2)

    metadata = {
        "render_modes": ["human", "rgb_array", "state_pixels"],
        "render_fps": FPS,
    }

    def __init__(self):
        pygame.init()
        self.contactListener_keepref = FrictionDetector(self)
        self.world = Box2D.b2World((0.0, -9.81), contactListener=self.contactListener_keepref)
        self.screen = None
        self.clock = None
        self.isopen = True

        # The wall
        self.wall = self.world.CreateBody(
            position=(0, 0),
            shapes=[b2EdgeShape(vertices=WALL_POLY[0:2]), b2EdgeShape(vertices=WALL_POLY[1:3]), b2EdgeShape(vertices=WALL_POLY[2:]), b2EdgeShape(vertices=[WALL_POLY[-1],WALL_POLY[-0]])],
            #shapes=b2LoopShape(vertices=WALL_POLY),
            shapeFixture=b2FixtureDef(
                density=1.0,
                restitution=0.9,  # 反弹系数（0~1）
                friction=0.1  # 降低摩擦增强反弹效果
            )
        )
        self.wall.mass = 10
        self.ball = Ball(self.world, init_x=COURT_W/2, init_y=TABLE_H)
        self.bat = Bat(self.world, init_x=COURT_W/2, init_y=(COURT_H - TABLE_H)/2)

        act_space_high = np.array(
            [
                self.bat_force_max,
                self.bat_angle_max,
            ],
            dtype=np.float32,
        )
        self.action_space = spaces.Box(-act_space_high, act_space_high, dtype=np.float32)

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
        self.t = 0.0

    def step(self, action):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.isopen = False
            if event.type == pygame.MOUSEMOTION:  # 鼠标控制球拍
                world_pos = screen_to_world(event.pos)
                self.bat.position = b2Vec2(world_pos)  # Y轴固定高度

        if action is not None:
            self.bat.act(action)

        self.ball.step()
        self.bat.step()
        self.world.Step(1.0 / self.FPS, 6 * 30, 2 * 30)
        self.t += 1.0 / self.FPS

        #self.state = self.render("state_pixels")

        return self.observation_space.sample(), 0, not self.isopen, {}

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

        if "t" not in self.__dict__:
            return

        self.surf = pygame.Surface((SCREEN_W, SCREEN_H))
        # 场地
        print(f'Court: {self.wall.position}')
        for i in range(4):
            pygame.draw.polygon(self.surf, color=(100, 100, 100), points=world_to_screen(self.wall.fixtures[i].shape.vertices), width=20)

        self.ball.draw(self.surf)
        self.bat.draw(self.surf)
        #self.surf = pygame.transform.flip(self.surf, False, True)

        font = pygame.font.Font(pygame.font.get_default_font(), 26)
        text = font.render("Text %04i" % 100, True, (255, 255, 255), (0, 0, 0))
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
            self.isopen = False

__all__ = ['PingPongEnv']
