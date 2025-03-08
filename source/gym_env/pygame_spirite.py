
import pygame, sys
import numpy as np
import math
from moviepy.video.tools.drawing import circle

from pygame.math import Vector2
from gym_env.b2drawer import B2Drawer

import Box2D
from Box2D import (b2Vec2, b2CircleShape, b2FixtureDef, b2LoopShape, b2PolygonShape,
                   b2RevoluteJointDef, b2PrismaticJointDef, b2_pi, b2WheelJointDef)
from gym_env.utils import world_to_screen, screen_to_world, FPS, PPM, COURT_W
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
        self.hull.mass = 2.75 / 1000

    def step(self):
        self.init_x = self.hull.position.x
        self.init_y = self.hull.position.y
        #print('ball w=\t%s\t%s' % world_to_screen((self.init_x, self.init_y)))

    def draw(self, surface, translation:tuple[int,int]=(0,0), angle:np.float32=None):
        #center = world_to_screen(Vector2(self.init_x, self.init_y) + Vector2(translation))
        #pygame.draw.circle(surface, color=(255, 255, 255), center=center, radius=self.radius * PPM)
        B2Drawer.draw_body(self.hull, surface)

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
            angularDamping=10,
            linearDamping=0.1,
            fixtures=[
                b2FixtureDef(
                    shape=b2PolygonShape(
                        vertices=self.BAT_POLY
                    ),
                    # 必须设置密度，否则ApplyTorque施加扭矩不转动
                    density=2.0,
                    restitution=0.1,  # 弹性提升
                    friction=0.3  # 摩擦降低
                )
            ])
        # 球拍质量约200g
        self.hull.mass=0.2
        #self.hull.fixedRotation = False

        # 创建y轴关节，只允许x轴移动
        # 定义静态刚体 anchor_body
        anchor_body = world.CreateStaticBody(
            position=(init_x, init_y),  # 设置锚点位置（例如坐标系原点）
            shapes=b2PolygonShape(box=(0.01, 0.01)), # 定义一个微小碰撞形状（避免无形状报错）
            userData="anchor", # 可选标识
        )
        for fixture in anchor_body.fixtures:
            fixture.sensor = True # 设置为感应器，不发生物理碰撞

        self.anchor_body = anchor_body
        target_body = self.hull
        # 此处用wheel关节，限制为x移动，可以旋转。不能使用棱柱关节b2PrismaticJointDef，不能转动
        joint_def = b2WheelJointDef()
        joint_def.Initialize(anchor_body, target_body, anchor_body.worldCenter, (1, 0))  # 滑动轴为 X 轴
        joint_def.enableLimit = True                        # 启用位移限制
        joint_def.lowerTranslation = -COURT_W / 2           # X 轴最小位移（左移范围）
        joint_def.upperTranslation = COURT_W / 2            # X 轴最大位移（右移范围）, 场地的宽度m

        # 关节移动的阻尼，不设置
        #joint_def.enableSpring = True
        #joint_def.stiffness = 10.0
        #joint_def.damping = 2.0

        world.CreateJoint(joint_def)

    def act(self, action):
        """
        球拍动作
        :param action:
        :return:
        """
        force = action[0]
        angle = 60 #action[1]
        force = (float(force), 0) #self.hull.GetWorldVector((float(force), 0))
        self.hull.ApplyForce(force, self.hull.worldCenter, True)

        # 用2pi取模
        angle_diff = (math.radians(angle) - self.hull.angle) % (2 * math.pi)
        # 限制在 [-pi ,pi] 之间
        angle_diff = angle_diff - 2 * math.pi if angle_diff > math.pi else angle_diff
        torque = pid.compute(angle_diff, 1.0 / FPS)
        self.hull.ApplyTorque(torque, True)

        # 角速度限制实现
        max_angular_speed = 0.1
        if abs(self.hull.angularVelocity) > max_angular_speed:
            self.hull.angularVelocity = math.copysign(max_angular_speed, self.hull.angularVelocity)

        print('bat.act \t center: %s, \t force: %s \t angle: %s \t angle_diff: %s \t torque: %s' % (self.hull.worldCenter, force, angle, angle_diff, torque))

        # 给球拍施加扭矩，旋转角度，限制角度范围
        """current_angle = self.hull.angle
        target_angle = math.radians(angle)  # 目标角度转弧度
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
        #print('bat w=\t%s\t%s    angel=\t角度:%s\t角速度:%s' % (pos[0], pos[1],  self.hull.angle, self.hull.angularVelocity))

    def draw(self, surface, translation:tuple[int,int]=(0,0), angle:np.float32=None):
        """
        points =  [
            Vector2(c).rotate(math.degrees(self.hull.angle)) + Vector2(self.hull.position) + Vector2(translation) for c in self.BAT_POLY
        ]
        pygame.draw.polygon(surface, color=(255, 255, 255), points=world_to_screen(points))
        """
        B2Drawer.draw_body(self.anchor_body, surface)
        B2Drawer.draw_body(self.hull, surface)

class PIDController:
    def __init__(self, kp, ki, kd):
        self.kp, self.ki, self.kd = kp, ki, kd
        self.last_error = self.integral = 0

    def compute(self, error, dt):
        self.integral += error * dt
        derivative = (error - self.last_error) / dt
        output = self.kp * error + self.ki * self.integral + self.kd * derivative
        self.last_error = error
        return output

# 初始化 PID 参数（根据刚体质量调试）
pid = PIDController(kp=0.1, ki=0, kd=0)

__all__ = ['Ball', 'Bat']