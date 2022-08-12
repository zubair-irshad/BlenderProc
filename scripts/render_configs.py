import numpy as np

IMG_WIDTH = 640
IMG_HEIGHT = 480

K = np.array([
            [400, 0, 320],
            [0, 400, 240],
            [0, 0, 1]
            ])

CAMERA_LOCS = {0: [[0, 2.5], [1.4, -0.8]], 
               1: [[-1.5, 2]],#, [0, 1.5], [-3, -2]], 
               2: [[1, 0], [0, 0]],
               3: [[2, 0], [-1.5, 2.5]],
               4: [[2, 1], [2.5, -1]]}

# OBJ_BAN_LIST is a global keyword ban list for all the rooms 
OBJ_BAN_LIST = ['Baseboard', 'Pocket', 'Floor', 'SlabSide.', 'WallInner', 'Front', 
                'WallTop', 'WallBottom', 'Ceiling.', 'FeatureWall', 'LightBand']

import yaml
ROOM_CONFIG = {}
with open('./scripts/room_configs.yaml', 'r') as f:
    ROOM_CONFIG = yaml.load(f, Loader=yaml.FullLoader)