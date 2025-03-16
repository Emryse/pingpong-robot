
import numpy as np
import math
from gym_env.b2drawer import B2Drawer
from Box2D import (b2Vec2, b2EdgeShape, b2CircleShape, b2FixtureDef, b2LoopShape, b2PolygonShape,
                   b2RevoluteJointDef, b2PrismaticJointDef, b2_pi, b2WheelJointDef)
from gym_env.utils import world_to_screen, screen_to_world, FPS, PPM, COURT_W

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

class Ball(BaseSprite):
    """
    Ping pong ball
    直径40mm, 重量2.75g
    """
    def __init__(self, world, init_x=0, init_y=0, init_angle:np.float32=0):
        super().__init__(world, init_x, init_y, init_angle)
        self.radius = 0.02 # 乒乓球标准直径 40mm
        self.body = self.world.CreateDynamicBody(
            fixtures=b2FixtureDef(
                shape=b2CircleShape(radius=self.radius),
                #density=1,
                restitution=0.9,  # 弹性提升
                friction=0.1,  # 摩擦降低
            ),
            bullet=True,
            position = (init_x, init_y),
            userData="ball",
        )
        self.body.damping = 0.15, # 空气阻力
        self.body.mass = 2.75 / 1000

    def step(self):
        self.init_x = self.body.position.x
        self.init_y = self.body.position.y
        #print('ball w=\t%s\t%s' % world_to_screen((self.init_x, self.init_y)))

    def draw(self, surface, translation:tuple[int,int]=(0,0), angle:np.float32=None):
        #center = world_to_screen(Vector2(self.init_x, self.init_y) + Vector2(translation))
        #pygame.draw.circle(surface, color=(255, 255, 255), center=center, radius=self.radius * PPM)
        B2Drawer.draw_body(self.body, surface)

class Bat(BaseSprite):
    """
    标准球拍尺寸 158*152mm 厚度15mm 约80-100g
    """
    # 球拍外轮廓
    BAT_THICKNESS =  0.015
    BAT_LENGTH =  0.158
    BAT_POLY = [(-0.079, -0.0075), (0.079, -0.0075), (0.079, 0.0075), (-0.079, 0.0075)]

    def __init__(self, world, init_x=0, init_y=0, init_angle:np.float32=0):
        BaseSprite.__init__(self, world, init_x, init_y, init_angle)
        # 球拍 Box2D动态刚体
        self.body = self.world.CreateDynamicBody(
            position=(init_x, init_y),
            angle=init_angle,
            angularDamping=10,
            linearDamping=0.1,
            fixtures=[
                b2FixtureDef(
                    shape=b2PolygonShape(
                        #vertices=self.BAT_POLY,
                        box=(self.BAT_LENGTH, self.BAT_THICKNESS),
                    ),
                    # 必须设置密度，否则ApplyTorque施加扭矩不转动
                    density=2.0,
                    restitution=0.1,  # 弹性提升
                    friction=0.3  # 摩擦降低
                )
            ],
            userData="bat",
        )
        # 球拍质量约200g
        self.body.mass=0.2

        # 创建y轴关节，只允许x轴移动
        # 虚拟的球拍X轴滑动轴
        x_axis_body = world.CreateStaticBody(
            position=(init_x, init_y),  # 设置锚点位置（例如坐标系原点）
            shapes=b2PolygonShape(box=(0.01, 0.01)),  # 定义一个微小碰撞形状（避免无形状报错）
            userData="x_axis_body",  # 可选标识
            # isSensor=True,         # 设置为感应器，不发生物理碰撞
        )
        for fixture in x_axis_body.fixtures:
            fixture.sensor = True  # 设置为感应器，不发生物理碰撞
        self.x_axis_body = x_axis_body

        slide_body = world.CreateDynamicBody(
            position=(init_x, init_y),  # 设置锚点位置（例如坐标系原点）
            shapes=b2PolygonShape(box=(0.01, 0.01)),  # 定义一个微小碰撞形状（避免无形状报错）
            userData="slide_body",  # 可选标识
            # isSensor=True,         # 设置为感应器，不发生物理碰撞
        )
        for fixture in slide_body.fixtures:
            fixture.sensor = True  # 设置为感应器，不发生物理碰撞
        self.slide_body = slide_body

        # 创建x轴棱柱滑动关节
        self.world.CreatePrismaticJoint(
            bodyA=x_axis_body,
            bodyB=slide_body,
            anchor=x_axis_body.worldCenter,
            axis=(1, 0),
            enableLimit=True,
            lowerTranslation=-COURT_W / 2,  # X 轴最小位移（左移范围）
            upperTranslation=COURT_W / 2  # X 轴最大位移（右移范围）, 场地的宽度/2
        )

        # 球拍下端铰链关节
        self.world.CreateRevoluteJoint(
            bodyA=slide_body,
            bodyB=self.body,
            anchor=self.body.position,
            lowerAngle=math.radians(-90),
            upperAngle=math.radians(90),
            enableLimit=True,
        )

        # 球拍扭矩和转动 PID控制参数（根据刚体质量调试）
        self.pid = PIDController(kp=0.01, ki=2, kd=0.00008)

    def act(self, action):
        """
        球拍动作
        :param action:
        :return:
        """
        force = action[0]
        angle = action[1]
        force = (float(force), 0) #self.hull.GetWorldVector((float(force), 0))
        self.body.ApplyForce(force, self.body.worldCenter, True)

        # 用2pi取模
        angle_diff = (math.radians(angle) - self.body.angle) % (2 * math.pi)
        # 限制在 [-pi ,pi] 之间
        angle_diff = angle_diff - 2 * math.pi if angle_diff > math.pi else angle_diff
        torque = self.pid.compute(angle_diff, 1.0 / FPS)
        self.body.ApplyTorque(torque, True)

        # 角速度限制实现
        max_angular_speed = 0.1
        if abs(self.body.angularVelocity) > max_angular_speed:
            self.body.angularVelocity = math.copysign(max_angular_speed, self.body.angularVelocity)

        #print('bat.act \t center: %s, \t force: %s \t angle: %s \t angle_diff: %s \t torque: %s' % (self.body.worldCenter, force, angle, angle_diff, torque))

        # 给球拍施加扭矩，旋转角度，限制角度范围
        """current_angle = self.body.angle
        target_angle = math.radians(angle)  # 目标角度转弧度
        angle_diff = target_angle - current_angle
        # 规范角度差到[-π, π]
        angle_diff = (angle_diff + math.pi) % (2 * math.pi) - math.pi

        torque_strength = 100.0  # 扭矩强度系数
        torque = angle_diff * torque_strength
        self.body.ApplyTorque(torque, wake=True)  # 施加扭矩‌:ml-citation{ref="2" data="citationList"}

        angular_damping = 0.5  # 阻尼系数(0-1)
        self.body.angularVelocity *= (1 - angular_damping)

        # 角速度限制实现
        max_angular_speed = 5.0  # 最大5弧度/秒
        if abs(self.body.angularVelocity) > max_angular_speed:
            self.body.angularVelocity = math.copysign(max_angular_speed, self.body.angularVelocity)
        """

    def step(self):
        self.init_x = self.body.position.x
        self.init_y = self.body.position.y
        pos = world_to_screen((self.init_x, self.init_y))
        #print('bat w=\t%s\t%s    angel=\t角度:%s\t角速度:%s' % (pos[0], pos[1],  self.body.angle, self.hull.angularVelocity))

    def draw(self, surface, translation:tuple[int,int]=(0,0), angle:np.float32=None):
        """
        points =  [
            Vector2(c).rotate(math.degrees(self.body.angle)) + Vector2(self.hull.position) + Vector2(translation) for c in self.BAT_POLY
        ]
        pygame.draw.polygon(surface, color=(255, 255, 255), points=world_to_screen(points))
        """
        B2Drawer.draw_body(self.x_axis_body, surface)
        B2Drawer.draw_body(self.slide_body, surface)
        B2Drawer.draw_body(self.body, surface)

class PIDController:
    """
    PID控制器
    """
    def __init__(self, kp, ki, kd):
        self.kp, self.ki, self.kd = kp, ki, kd
        self.last_error = self.integral = 0

    def compute(self, error, dt):
        self.integral += error * dt
        derivative = (error - self.last_error) / dt
        output = self.kp * error + self.ki * self.integral + self.kd * derivative
        self.last_error = error
        return output

__all__ = ['Ball', 'Bat']