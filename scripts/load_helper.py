import numpy as np
import os

import blenderproc as bproc
from blenderproc.python.types.MeshObjectUtility import MeshObject
import re
import bpy
import json

# LAYOUT_DIR = '/data/yliugu/3D-FRONT'
# TEXTURE_DIR = '/data/yliugu/3D-FRONT-texture'
# MODEL_DIR = '/data/yliugu/3D-FUTURE-model'

LAYOUT_DIR = "/wild6d_data/zubair/3DFRONT_Raw/3D-FRONT"
TEXTURE_DIR = "/wild6d_data/zubair/3DFRONT_Raw/3D-FRONT-texture"
MODEL_DIR = "/wild6d_data/zubair/3DFRONT_Raw/3D-FUTURE-model"

# LAYOUT_DIR = "/home/mirshad7/Downloads/3D-FRONT"
# TEXTURE_DIR = "/home/mirshad7/Downloads/3D-FRONT-texture"
# MODEL_DIR = "/home/mirshad7/Downloads/3D-FUTURE-model"

RENDER_TEMP_DIR = "./FRONT3D_render/temp"
# SCENE_LIST = []


def check_cache_dir(scene_idx):
    if not os.path.isdir(f"./cached/{scene_idx}"):
        os.makedirs(f"./cached/{scene_idx}")


def get_scene_rot_bbox_meta(scene_idx, overwrite=False):
    """Get the bounding box meta data of a scene.
    [(name1, [[xmin, ymin, zmin], [xmax, ymax, zmax]]), (name2, [[xmin, ymin, zmin], [xmax, ymax, zmax]]), ...]
    """
    check_cache_dir(scene_idx)
    if os.path.isfile("./cached/%d/bboxes.npy" % scene_idx) and overwrite == False:
        print(f"Found cached information for scene {scene_idx}.")
        names = np.load(f"./cached/{scene_idx}/names.npy")
        # with open('./cached/{scene_idx}/bbox.json', 'r') as f:
        #     bbox = json.load(f)
        bboxes = np.load(f"./cached/{scene_idx}/bboxes.npy")

    else:
        loaded_objects = load_scene_objects(scene_idx, overwrite)
        names = []
        bboxes = []

        for i in range(len(loaded_objects)):
            object = loaded_objects[i]
            name = object.get_name()
            bbox = object.get_bound_box()

            names.append(name)
            bboxes.append(bbox)

        np.save(f"./cached/{scene_idx}/names.npy", names)
        np.save(f"./cached/{scene_idx}/bboxes.npy", bboxes)

    return names, bboxes


def get_scene_bbox_meta(scene_idx, overwrite=False):
    """Get the bounding box meta data of a scene.
    [(name1, [[xmin, ymin, zmin], [xmax, ymax, zmax]]), (name2, [[xmin, ymin, zmin], [xmax, ymax, zmax]]), ...]
    """
    check_cache_dir(scene_idx)
    if os.path.isfile("./cached/%d/bbox_mins.npy" % scene_idx) and overwrite == False:
        print(f"Found cached information for scene {scene_idx}.")
        names = np.load(f"./cached/{scene_idx}/names.npy")
        bbox_mins = np.load(f"./cached/{scene_idx}/bbox_mins.npy")
        bbox_maxs = np.load(f"./cached/{scene_idx}/bbox_maxs.npy")
    else:
        loaded_objects = load_scene_objects(scene_idx, overwrite)
        names = []
        bbox_mins = []
        bbox_maxs = []

        print("=============================================\n\n\n\n")
        print("OBJECTSSSSSS")
        for i in range(len(loaded_objects)):
            object = loaded_objects[i]
            name = object.get_name()
            print("name", name)
            bbox = object.get_bound_box()
            print("bbox", bbox)
            bbox_min = np.min(bbox, axis=0)
            bbox_max = np.max(bbox, axis=0)
            names.append(name)
            bbox_mins.append(bbox_min)
            bbox_maxs.append(bbox_max)

        print("==================================================\n\n\n\n")

        np.save(f"./cached/{scene_idx}/names.npy", names)
        np.save(f"./cached/{scene_idx}/bbox_mins.npy", bbox_mins)
        np.save(f"./cached/{scene_idx}/bbox_maxs.npy", bbox_maxs)

    return names, bbox_mins, bbox_maxs, loaded_objects


def add_texture(obj: MeshObject, tex_path):
    """Add a texture to an object."""
    obj.clear_materials()
    mat = obj.new_material("my_material")
    bsdf = mat.nodes["Principled BSDF"]
    texImage = mat.nodes.new("ShaderNodeTexImage")
    texImage.image = bpy.data.images.load(tex_path)
    mat.links.new(bsdf.inputs["Base Color"], texImage.outputs["Color"])


# TODO: read config file
def load_scene_objects(scene_idx, scene_list_all=None):
    check_cache_dir(scene_idx)
    mapping_file = bproc.utility.resolve_resource(
        os.path.join("front_3D", "3D_front_mapping.csv")
    )
    mapping = bproc.utility.LabelIdMapping.from_csv(mapping_file)

    loaded_objects = bproc.loader.load_front3d(
        json_path=scene_list_all[scene_idx],
        future_model_path=MODEL_DIR,
        front_3D_texture_path=TEXTURE_DIR,
        label_mapping=mapping,
        ceiling_light_strength=1,
        lamp_light_strength=30,
    )

    # add texture to wall and floor. Otherwise they will be white.
    for obj in loaded_objects:
        name = obj.get_name()
        if "wall" in name.lower():
            add_texture(
                obj, TEXTURE_DIR + "/1b57700d-f41b-4ac7-a31a-870544c3d608/texture.png"
            )
        elif "floor" in name.lower():
            add_texture(
                obj, TEXTURE_DIR + "/0b48b46d-4f0b-418d-bde6-30ca302288e6/texture.png"
            )
        # elif 'ceil' in name.lower():

    return loaded_objects


mapping_file = bproc.utility.resolve_resource(
    os.path.join("front_3D", "3D_front_mapping.csv")
)
mapping = bproc.utility.LabelIdMapping.from_csv(mapping_file)


def load_scene_objects_wotexture(scene_idx, scene_list_all):
    check_cache_dir(scene_idx)
    # mapping_file = bproc.utility.resolve_resource(
    #     os.path.join("front_3D", "3D_front_mapping.csv")
    # )

    loaded_objects = bproc.loader.load_front3d(
        json_path=scene_list_all[scene_idx],
        future_model_path=MODEL_DIR,
        front_3D_texture_path=TEXTURE_DIR,
        label_mapping=mapping,
        ceiling_light_strength=1,
        lamp_light_strength=30,
    )

    # # add texture to wall and floor. Otherwise they will be white.
    # for obj in loaded_objects:
    #     name = obj.get_name()
    #     if "wall" in name.lower():
    #         add_texture(
    #             obj, TEXTURE_DIR + "/1b57700d-f41b-4ac7-a31a-870544c3d608/texture.png"
    #         )
    #     elif "floor" in name.lower():
    #         add_texture(
    #             obj, TEXTURE_DIR + "/0b48b46d-4f0b-418d-bde6-30ca302288e6/texture.png"
    #         )
    #     # elif 'ceil' in name.lower():

    return loaded_objects
