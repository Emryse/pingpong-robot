
import numpy as np
from pygame.math import Vector2
from Box2D import (b2Vec2)

# 像素/米（4K优化）
PPM = 100
# 球桌宽高m
TABLE_W, TABLE_H = 1.525, 2.74
# 场地尺寸
COURT_W, COURT_H = TABLE_W + 3, TABLE_H + 3
# 屏幕显示
SCREEN_W, SCREEN_H = COURT_W * PPM, COURT_H * PPM

def world_to_screen(pos):
    if isinstance(pos, (tuple, b2Vec2, Vector2)) or (isinstance(pos, np.ndarray) and pos.shape == (2,)):
        return int(pos[0] * PPM), SCREEN_H - int(pos[1] * PPM)
    if isinstance(pos,  (list, np.ndarray)):
        return [world_to_screen(p) for p in pos]
    return pos

def screen_to_world(pos):
    return int(pos[0] / PPM), SCREEN_H / PPM - int(pos[1] / PPM)