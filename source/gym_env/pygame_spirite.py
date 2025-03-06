
import pygame, sys
import numpy as np
import math
from moviepy.video.tools.drawing import circle

from pygame.math import Vector2
from pygame import gfxdraw

import Box2D
from Box2D import (b2Vec2, b2CircleShape, b2FixtureDef, b2LoopShape, b2PolygonShape,
                   b2RevoluteJointDef, b2_pi)
from gym_env.utils import world_to_screen, screen_to_world, PPM
from pingpong import world


class BaseSprite:

    """
    属性 物理世界单位 m
    init_x: np.float32
    init_y: np.float32
    # 旋转角度
    angle: np.float32

    # 质量
    mass: np.float32
    # 当前速度，向量
    velocity: np.ndarray
    # 当前受力，向量
    force: np.ndarray

    """

    def __init__(self, world, init_x=0, init_y=0, init_angle:np.float32=0):
        #super().__init__(self)
        self.world = world
        self.init_x = init_x
        self.init_y = init_y
        self.angle = init_angle

    def step(self):
        """
        更新状态，为绘图作准备
        """
        pass

    def draw(self, surface, translation:tuple[int,int]=(0,0), angle:np.float32=None):
        """
        绘图
        """
        pass

    def collide(self, other) -> bool:
        """
        碰撞检测
        """
        return False

    @property
    def outline(self) -> np.ndarray:
        return np.ndarray()


class Ball(BaseSprite):
    """
    Ping pong ball
    直径40mm, 重量2.75g
    """
    def __init__(self, world, init_x=0, init_y=0, init_angle:np.float32=0):
        super().__init__(world, init_x, init_y, init_angle)
        self.radius = 0.02 # 乒乓球标准直径 40mm
        self.hull = self.world.CreateDynamicBody(
            fixtures=b2FixtureDef(
                shape=b2CircleShape(radius=self.radius),
                #density=1,
                restitution=0.9,  # 弹性提升
                friction=0.1,  # 摩擦降低
            ),
            bullet=True,
            position = (init_x, init_y),
        )
        self.hull.damping = 0.15, # 空气阻力
        self.hull.mass = 2.75 / 10

    def step(self):
        self.init_x = self.hull.position.x
        self.init_y = self.hull.position.y
        #print('ball w=\t%s\t%s' % world_to_screen((self.init_x, self.init_y)))

    def draw(self, surface, translation:tuple[int,int]=(0,0), angle:np.float32=None):
        center = world_to_screen(Vector2(self.init_x, self.init_y) + Vector2(translation))
        pygame.draw.circle(surface, color=(255, 255, 255), center=center, radius=self.radius * PPM)

class Bat(BaseSprite):
    """
    标准球拍尺寸 158*152mm 厚度15mm 约80-100g
    """
    BAT_POLY = [(-0.079, -0.0075), (0.079, -0.0075), (0.079, 0.0075), (-0.079, 0.0075)]
    def __init__(self, world, init_x=0, init_y=0, init_angle:np.float32=0):
        BaseSprite.__init__(self, world, init_x, init_y, init_angle)
        self.hull = self.world.CreateDynamicBody(
            position=(init_x, init_y),
            angle=init_angle,
            angularDamping=5,
            linearDamping=0.1,
            fixtures=[
                b2FixtureDef(
                    shape=b2PolygonShape(
                        vertices=self.BAT_POLY
                    ),
                    #density=10.0,
                    restitution=0.1,  # 弹性提升
                    friction=0.1  # 摩擦降低
                )
            ])
        self.hull.mass=1

    def act(self, action):
        """
        球拍动作
        :param action:
        :return:
        """
        force = action[0]
        angle = action[1]
        force = self.hull.GetWorldVector(localVector=(float(0), self.hull.mass * 9.81))
        self.hull.ApplyForce(force, self.hull.worldCenter, True)
        #self.hull.angle = float(math.radians(45))
        self.hull.ApplyTorque(50, True)

        # 给球拍施加扭矩，旋转角度，限制角度范围
        """
        current_angle = self.hull.angle
        target_angle = math.radians(90)  # 目标角度转弧度
        angle_diff = target_angle - current_angle
        # 规范角度差到[-π, π]
        angle_diff = (angle_diff + math.pi) % (2 * math.pi) - math.pi

        torque_strength = 100.0  # 扭矩强度系数
        torque = angle_diff * torque_strength
        self.hull.ApplyTorque(torque, wake=True)  # 施加扭矩‌:ml-citation{ref="2" data="citationList"}

        angular_damping = 0.5  # 阻尼系数(0-1)
        self.hull.angularVelocity *= (1 - angular_damping)

        # 角速度限制实现
        max_angular_speed = 5.0  # 最大5弧度/秒
        if abs(self.hull.angularVelocity) > max_angular_speed:
            self.hull.angularVelocity = math.copysign(max_angular_speed, self.hull.angularVelocity)
        
        """

    def step(self):
        self.init_x = self.hull.position.x
        self.init_y = self.hull.position.y
        pos = world_to_screen((self.init_x, self.init_y))
        print('bat w=\t%s\t%s    angel=\t角度:%s\t角速度:%s' % (pos[0], pos[1],  self.hull.angle, self.hull.angularVelocity))

    def draw(self, surface, translation:tuple[int,int]=(0,0), angle:np.float32=None):
        points =  [
            Vector2(c).rotate(math.degrees(self.hull.angle)) + Vector2(self.hull.position) + Vector2(translation) for c in self.BAT_POLY
        ]
        pygame.draw.polygon(surface, color=(255, 255, 255), points=world_to_screen(points))


__all__ = ['Ball', 'Bat']