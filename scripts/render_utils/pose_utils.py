############################## poses generation ##################################
import numpy as np
from random import shuffle

pi = np.pi
cos = np.cos
sin = np.sin
COMPUTE_DEVICE_TYPE = "CUDA"


def normalize(x, axis=-1, order=2):
    l2 = np.linalg.norm(x, order, axis)
    l2 = np.expand_dims(l2, axis)
    l2[l2 == 0] = 1
    return (x / l2,)


def look_at_rotation(camera_position, at=None, up=None, inverse=False, cv=False):
    """
    This function takes a vector 'camera_position' which specifies the location
    of the camera in world coordinates and two vectors `at` and `up` which
    indicate the position of the object and the up directions of the world
    coordinate system respectively. The object is assumed to be centered at
    the origin.
    The output is a rotation matrix representing the transformation
    from world coordinates -> view coordinates.
    Input:
        camera_position: 3
        at: 1 x 3 or N x 3  (0, 0, 0) in default
        up: 1 x 3 or N x 3  (0, 1, 0) in default
    """

    if at is None:
        at = np.zeros_like(camera_position)
    else:
        at = np.array(at)
    if up is None:
        up = np.zeros_like(camera_position)
        up[2] = -1
    else:
        up = np.array(up)

    z_axis = normalize(camera_position - at)[0]
    x_axis = normalize(np.cross(up, z_axis))[0]
    y_axis = normalize(np.cross(z_axis, x_axis))[0]

    R = np.concatenate([x_axis[:, None], y_axis[:, None], z_axis[:, None]], axis=1)
    return R


def c2w_from_loc_and_at(cam_pos, at, up=(0, 0, 1)):
    """Convert camera location and direction to camera2world matrix."""
    c2w = np.eye(4)
    cam_rot = look_at_rotation(cam_pos, at=at, up=up, inverse=False, cv=True)
    c2w[:3, 3], c2w[:3, :3] = cam_pos, cam_rot
    return c2w


def pos_in_bbox(pos, bbox):
    """
    Check if a point is inside a bounding box.
    Input:
        pos: 3 x 1
        bbox: 2 x 3
    Output:
        True or False
    """
    return (
        pos[0] >= bbox[0][0]
        and pos[0] <= bbox[1][0]
        and pos[1] >= bbox[0][1]
        and pos[1] <= bbox[1][1]
        and pos[2] >= bbox[0][2]
        and pos[2] <= bbox[1][2]
    )


def check_pos_valid(pos, room_objs_dict, room_bbox):
    """Check if the position is in the room, not too close to walls and not conflicting with other objects."""
    room_bbox_small = [
        [item + 0.5 for item in room_bbox[0]],
        [room_bbox[1][0] - 0.5, room_bbox[1][1] - 0.5, room_bbox[1][2] - 0.8],
    ]  # ceiling is lower
    if not pos_in_bbox(pos, room_bbox_small):
        return False
    for obj_dict in room_objs_dict["objects"]:
        obj_bbox = obj_dict["aabb"]
        if pos_in_bbox(pos, obj_bbox):
            return False

    return True


def generate_room_poses(
    scene_idx,
    room_idx,
    room_objs_dict,
    room_bbox,
    num_poses_per_object,
    max_global_pos,
    global_density,
    room_config,
):
    """Return a list of poses including global poses and close-up poses for each object."""

    poses = []
    num_closeup, num_global = 0, 0
    h_global = 1.2

    # close-up poses for each object.
    if num_poses_per_object > 0:
        for obj_dict in room_objs_dict["objects"]:
            obj_bbox = np.array(obj_dict["aabb"])
            cent = np.mean(obj_bbox, axis=0)
            rad = (
                np.linalg.norm(obj_bbox[1] - obj_bbox[0]) / 2 * 1.7
            )  # how close the camera is to the object
            if np.max(obj_bbox[1] - obj_bbox[0]) < 1:
                rad *= 1.2  # handle small objects

            positions = []
            n_hori_sects = 30
            n_vert_sects = 10
            theta_bound = [0, 2 * pi]
            phi_bound = [-pi / 4, pi / 4]
            theta_sect = (theta_bound[1] - theta_bound[0]) / n_hori_sects
            phi_sect = (phi_bound[1] - phi_bound[0]) / n_vert_sects
            for i_vert_sect in range(n_vert_sects):
                for i_hori_sect in range(n_hori_sects):
                    theta_a = theta_bound[0] + i_hori_sect * theta_sect
                    theta_b = theta_a + theta_sect
                    phi_a = phi_bound[0] + i_vert_sect * phi_sect
                    phi_b = phi_a + phi_sect
                    theta = np.random.uniform(theta_a, theta_b)
                    phi = np.random.uniform(phi_a, phi_b)
                    pos = [cos(phi) * cos(theta), cos(phi) * sin(theta), sin(phi)]
                    positions.append(pos)
            positions = np.array(positions)
            positions = positions * rad + cent

            positions = [
                pos
                for pos in positions
                if check_pos_valid(pos, room_objs_dict, room_bbox)
            ]
            shuffle(positions)
            if len(positions) > num_poses_per_object:
                positions = positions[:num_poses_per_object]

            poses.extend([c2w_from_loc_and_at(pos, cent) for pos in positions])

            num_closeup = len(positions)

    # global poses
    if max_global_pos > 0:
        bbox = room_config[scene_idx][room_idx]["bbox"]
        x1, y1, x2, y2 = bbox[0][0], bbox[0][1], bbox[1][0], bbox[1][1]
        rm_cent = np.array([(x1 + x2) / 2, (y1 + y2) / 2, h_global])

        # flower model
        rad_bound = [0.3, 5]
        rad_intv = global_density
        theta_bound = [0, 2 * pi]
        theta_sects = 20
        theta_intv = (theta_bound[1] - theta_bound[0]) / theta_sects
        h_bound = [0.8, 2.0]

        positions = []
        theta = theta_bound[0]
        for i in range(theta_sects):
            rad = rad_bound[0]
            while rad < rad_bound[1]:
                h = np.random.uniform(h_bound[0], h_bound[1])
                pos = [rm_cent[0] + rad * cos(theta), rm_cent[1] + rad * sin(theta), h]
                if check_pos_valid(pos, room_objs_dict, room_bbox):
                    positions.append(pos)
                rad += rad_intv
            theta += theta_intv
        positions = np.array(positions)
        np.random.shuffle(positions)

        if len(positions) > max_global_pos:
            positions = positions[:max_global_pos]

        poses.extend(
            [
                c2w_from_loc_and_at(pos, [rm_cent[0], rm_cent[1], pos[2]])
                for pos in positions
            ]
        )

        num_global = len(positions)

    return poses, num_closeup, num_global


#################################################################################
