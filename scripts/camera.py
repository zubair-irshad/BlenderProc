import numpy as np

CAMERA_LOCS = {0: [[0, 2.5], [1.4, -0.8]], 
               1: [[-1.5, 2]],#, [0, 1.5], [-3, -2]], 
               2: [[1, 0], [0, 0]],
               3: [[2, 0], [-1.5, 2.5]],
               4: [[2, 1], [2.5, -1]]}
# center: the center of the ellipse
# a: the semi-major axis, b: the semi-minor axis
# num_cam: the number of cameras
# bbox: the bounding box of the room
# corners: two of the corner camera locations
# Example:{'center': (, ), 'a': , 'b': , 'num_cam': 16, 'bbox': [[, ], [, ]]},
ROOM_CONFIG = {3: {0: {'center': (3, 2.2), 'a': 1.5, 'b': 2.2, 'num_cam': 16, 'bbox': [[0.8, -1.0], [5, 5]], 'corners': [[1.4, 0], [4.5, 4.3]]},
                   1: {'center': (3.3, -4.8), 'a': 1.3, 'b': 2, 'num_cam': 8, 'bbox': [[1.3, -7.35], [5.0, -2.5]]},
                   2: {'center': (-0.4, -5), 'a': 1.5, 'b': 2.1, 'num_cam': 16, 'bbox': [[-2.2, -7.35], [1.3, -2.5]]}},
               4: {0: {'center': (4.5, -2.3), 'a': 1.7, 'b': 1, 'num_cam': 16, 'bbox': [[1.5, -4], [6.5, -0.5]], }}}

K = np.array([
    [400, 0, 320],
    [0, 400, 240],
    [0, 0, 1]
])

IMG_WIDTH = 640
IMG_HEIGHT = 480