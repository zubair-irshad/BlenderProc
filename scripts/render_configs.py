import numpy as np
import yaml

IMG_WIDTH = 640
IMG_HEIGHT = 480

K = np.array([
            [400, 0, 320],
            [0, 400, 240],
            [0, 0, 1]
            ])

# OBJ_BAN_LIST is a global keyword ban list for all the rooms 
OBJ_BAN_LIST = ['Baseboard', 'Pocket', 'Floor', 'SlabSide.', 'WallInner', 'Front', 
                'WallTop', 'WallBottom', 'Ceiling.', 'FeatureWall', 'LightBand']

ROOM_CONFIG = {}
with open('./scripts/room_configs.yaml', 'r') as f:
    ROOM_CONFIG = yaml.load(f, Loader=yaml.FullLoader)