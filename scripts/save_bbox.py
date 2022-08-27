
import blenderproc as bproc
bproc.init(compute_device='cuda:0', compute_device_type='CUDA')
import sys
import json
sys.path.append('/home/yliugu/BlenderProc/scripts')

from render_configs import *
from render import load_scene_objects, get_room_objects, get_room_bbox, construct_scene_list, merge_bbox

# To run the script, 'python cli.py run ./scripts/save_bbox.py'

construct_scene_list()
# skip = {1000: [0]}
skip = {}


all_scene_keys = ROOM_CONFIG.keys()
all_skip_keys = skip.keys()
for cur_scene_id in range(1000,1001):
    if cur_scene_id not in all_scene_keys:
        print(f'Scene {cur_scene_id} does not have config.')
        continue
    scene_info = ROOM_CONFIG[cur_scene_id]
    scene_room_configs = scene_info.keys()

    for cur_room_id in scene_room_configs:
        if cur_scene_id in all_skip_keys:
            if cur_room_id in skip[cur_scene_id]:
                print(f'Skip room {cur_room_id} of scene {cur_scene_id}.')
                continue
        loaded_objects = load_scene_objects(cur_scene_id)
        room_objects = get_room_objects(cur_scene_id, cur_room_id, loaded_objects)
        room_bbox = get_room_bbox(cur_scene_id, cur_room_id, loaded_objects=loaded_objects)
        room_bbox_meta = []
        for obj in room_objects:
            obj_bbox_8 = obj.get_bound_box()
            obj_bbox = np.array([np.min(obj_bbox_8, axis=0), np.max(obj_bbox_8, axis=0)])
            room_bbox_meta.append((obj.get_name(), obj_bbox))

        room_bbox_meta = merge_bbox(cur_scene_id, cur_room_id, room_bbox_meta)
        result_room_bbox_meta = []
        for bbox_meta in room_bbox_meta:
            flag_use = True
            obj_name = bbox_meta[0]
            for ban_word in OBJ_BAN_LIST:
                if ban_word in obj_name:
                    flag_use=False
            if 'keyword_ban_list' in ROOM_CONFIG[cur_scene_id][cur_room_id].keys():
                for ban_word in ROOM_CONFIG[cur_scene_id][cur_room_id]['keyword_ban_list']:
                    if ban_word in obj_name:
                        flag_use=False
            if 'fullname_ban_list' in ROOM_CONFIG[cur_scene_id][cur_room_id].keys():
                for fullname in ROOM_CONFIG[cur_scene_id][cur_room_id]['fullname_ban_list']:
                    if fullname == obj_name.strip():
                        flag_use=False
            if flag_use:
                result_room_bbox_meta.append(bbox_meta)
        room_bbox_meta = result_room_bbox_meta
        room_bbox = np.array(room_bbox)
        scale = 1.5 / np.max(room_bbox[1] - room_bbox[0])
        cent_after_scale = scale * (room_bbox[0] + room_bbox[1])/2.0
        offset = np.array([0.5, 0.5, 0.5]) - cent_after_scale


        bbox_list = []
        for _, obj in enumerate(room_bbox_meta):
            obj_bbox = np.array(obj[1])
            obj_bbox_ngp = {
                "extents": (obj_bbox[1]-obj_bbox[0]).tolist(),
                "orientation": np.eye(3).tolist(),
                "position": ((obj_bbox[0]+obj_bbox[1])/2.0).tolist(),
            }
            bbox_list.append(obj_bbox_ngp)


        out = {}
        out['out_bounding'] = bbox_list
        with open(f'transforms_{cur_scene_id}_{cur_room_id}.json', 'w') as f:
            json.dump(out, f, indent=4)