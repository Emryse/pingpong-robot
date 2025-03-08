# -*- coding: utf-8 -*-
# 2025.02.22 ITTF物理模拟增强版
import numpy as np
import pygame
from Box2D import (b2World, b2CircleShape, b2FixtureDef, b2_staticBody,
                   b2Vec2, b2EdgeShape, b2PolygonShape, b2ContactListener)

# ███████ 核心参数 ████████
SCREEN_WIDTH, SCREEN_HEIGHT = 800, 600
PPM = 50  # 像素/米（4K优化）
GRAVITY = (0.0, -9.81)

# ███████ 物理世界初始化 ███████
world = b2World(gravity=GRAVITY)
world.linearDamping = 0.05  # 新增线性阻尼
world.angularDamping = 0.1  # 角速度阻尼
world.velocityThreshold = 1.0  # 激活速度阈值
world.restitutionThreshold = 0.5  # 弹性激活阈值

# ███████ 自定义碰撞监听器 ███████
class BounceDetector(b2ContactListener):
    def BeginContact(self, contact):
        # 提取碰撞双方标识
        fixA, fixB = contact.fixtureA, contact.fixtureB
        types = {fixA.userData, fixB.userData}

        # 精确匹配球与球拍组合
        if types == {"BALL", "PADDLE"}:
            ball = fixA.body if fixA.userData == "BALL" else fixB.body
            print(f"[有效碰撞] 垂直速度: {ball.linearVelocity.y:.2f} m/s")


contact_listener = BounceDetector()
world.contactListener = contact_listener


# ███████ 实体创建函数 ████████
def create_ball():
    ball_fixture = b2FixtureDef(
        userData='BALL',
        shape=b2CircleShape(radius=0.2),
        density=2.75 / 1000,
        restitution=0.9,  # 弹性提升
        friction=0.1  # 摩擦降低
    )
    body = world.CreateDynamicBody(position=(8, 10))  # 起始高度增加
    body.CreateFixture(ball_fixture)
    body.damping = 0.15 # 空气阻力
    return body


def create_paddle():
    body = world.CreateDynamicBody(position=(8, 5))  # 改为动态体
    body.CreateFixture(
        userData='PADDLE',
        shape=b2PolygonShape(box=(5, 0.1)),  # 改用矩形碰撞体
        density=1.0,
        restitution=0.85
    )
    body.type = b2_staticBody  # 保持静止状态
    return body


# ███████ 坐标转换系统 ████████
def world_to_screen(pos):
    if isinstance(pos, (tuple, b2Vec2)) or (isinstance(pos, np.ndarray) and pos.shape == (2,)):
        return int(pos[0] * PPM), SCREEN_HEIGHT - int(pos[1] * PPM)
    if isinstance(pos,  (list, np.ndarray)):
        return [world_to_screen(p) for p in pos]
    return pos

def screen_to_world(pos):
    return int(pos[0] / PPM), SCREEN_HEIGHT / PPM - int(pos[1] / PPM)

# ███████ 主程序 ██████████
def main():
    pygame.init()
    screen = pygame.display.set_mode(
        (SCREEN_WIDTH, SCREEN_HEIGHT),
        pygame.SCALED | pygame.HWSURFACE,
        vsync=1
    )
    clock = pygame.time.Clock()

    ball = create_ball()
    paddle = create_paddle()
    running = True

    while running:
        # ███ 事件处理 ███
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.MOUSEMOTION:  # 鼠标控制球拍
                world_pos = screen_to_world(event.pos)
                paddle.position = b2Vec2(world_pos)  # Y轴固定高度

        # ███ 物理更新 ███
        world.Step(1 / 60, 12, 8)

        # ███ 渲染逻辑 ███
        screen.fill((25, 25, 35))  # 深空背景色

        # 绘制球拍
        paddle_pos = world_to_screen(paddle.position)
        paddle_points = world_to_screen(np.array(paddle.fixtures[0].shape.vertices) + np.array([paddle.position.x, paddle.position.y]))
        pygame.draw.polygon(screen, (200, 200, 255),
                         points = paddle_points)

        # 绘制乒乓球
        ball_pos = world_to_screen(ball.position)
        radius = int(0.2 * PPM)
        pygame.draw.circle(screen, (255, 80, 80),
                           ball_pos, radius)

        pygame.display.flip()
        clock.tick(60)

        # 自动停止检测
        if ball.linearVelocity.length < 0.0001:  # 速度阈值
             print("Ball has stopped.")
             running = False

    pygame.quit()


if __name__ == "__main__":
    main()
