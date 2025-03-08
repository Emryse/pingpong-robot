import Box2D
import pygame
from Box2D import b2PolygonShape, b2CircleShape, b2EdgeShape
from pygame.surface import SurfaceType
from .utils import world_to_screen, PPM

class B2Drawer:
    """
    Box2BBody的通用绘制类
    """
    def __init__(self, ):
        pass

    @staticmethod
    def draw_body(body:Box2D.b2Body, screen:SurfaceType):
        for fixture in body.fixtures:
            shape = fixture.shape
            # 根据形状类型调用不同绘制方法
            if isinstance(shape, b2PolygonShape):
                B2Drawer.__draw_polygon(body, screen, shape)
            elif isinstance(shape, b2CircleShape):
                B2Drawer.__draw_circle(body, screen, shape)
            elif isinstance(shape, b2EdgeShape):
                B2Drawer.__draw_edge(body, screen, shape)

    @staticmethod
    def __draw_polygon(body, screen:SurfaceType, shape):
        vertices = [body.transform * v for v in shape.vertices]  # 局部坐标转世界坐标
        screen_points = [world_to_screen(v) for v in vertices]
        pygame.draw.polygon(screen, (255,0,0), screen_points, width=2)  # 绘制闭合多边形

    @staticmethod
    def __draw_circle(body, screen:SurfaceType, shape):
        center = body.transform * shape.pos  # 圆心世界坐标
        radius_px = shape.radius * PPM       # 半径转像素
        screen_center = world_to_screen(center)
        pygame.draw.circle(screen, (0,255,0), screen_center, int(radius_px), width=2)

    @staticmethod
    def __draw_edge(body, screen:SurfaceType, shape):
        v1 = body.transform * shape.vertices  # 起点世界坐标
        v2 = body.transform * shape.vertices  # 终点世界坐标
        p1 = world_to_screen(v1)
        p2 = world_to_screen(v2)
        pygame.draw.line(screen, (0,0,255), p1, p2, width=2)
