# 3D Front Dataset rendering

## Need to know
- The 3DFRONT dataset is saved at:
  - Scene setting config files: `/data2/jhuangce/3D-FRONT`
  - Textures: `/data2/jhuangce/3D-FRONT-texture`
  - Models: `/data2/jhuangce/3D-FRONT-model`
- There are **6813 scenes** in 3DFRONT dataset. Each scene is a suite that can be divided into several single rooms (several single nerf scenes).
- So, each time of rendering, you need to specify a `scene_idx` and a `room_idx`. 

## Setup
- Install Blenderproc via pip
  ``` bash
  pip install blenderproc
  ```

## Steps for rendering 3DFRONT
1. Generate the floor plan of a new scene. \
   Run the following command to generate the floor plan of scene 25. For convenience you can specify the room number by `-r 0` even though you haven't configure any room in this scene. (You will configure rooms in the next step.)
   ``` shell
   python cli.py run ./scripts/render.py -s 25 -r 0 --plan
   ```
   Then, you can find the floor plan of scene 25 at  `FRONT3D_render/3dfront_0025_00/overview/floor_plan.jpg`
2. Configure a room in the scene. \
   Room configuration should be done in `scripts/room_config.yml` \
   The first level key of `ROOM_CONFIG` is the scene index, the second level key is the room index. You can create any number of rooms under one scene. For now, only fill the `bbox` in format `[[xmin, ymin], [xmax, ymax]]`, and leave `keyword_ban_list` and `fullname_ban_list` as empty list. Carefully refer to the floor plan that you generated to determine the bounding box of each room. \
   After defining `bbox` for a room, you can generate the floor plan image again to see whether the bounding box of the room is actually what you expected. 
   
3. Generate an overview of the scene. \
   Run the following command to render 4 overview pictures of scene 25 room 0 to check whether this room is really useful. 
   ``` bash
   python cli.py run ./scripts/render.py --gpu 1 -s 25 -r 0 --overview
   ```
   After rendering, you can find the overview images in `FRONT3D_render/3dfront_0025_00/overview/`
   - If the room is open connected to another room (for example a living room can be connected to another living room without a wall in between, even though they look like two seperate rooms in the floor plan image), then don't use it.
   - If the room is valid, then carefully ban those unecessary objects in the scene by updating `keyword_ban_list` and `fullname_ban_list`. You can also merge several bboxes into one large bbox by adding a list into `merge_list`. 
     - If an object label **contains any item** in the `keyword_ban_list`, the object bounding box will not be used. 
     - If an object label **appears** in the `fullname_ban_list`, the object bounding box will not be used. \
     - Any bounding box list in `merge_list` will be merged into one bounding box. 
   Only keep those large objects. (Usually there will be <10 large objects in a single room. ) Render the overview images again to make sure only those desired bounding boxes appear.

4. Render the room. \
   Run the following command to render scene 25 room 0. 
   ``` bash
   python cli.py run ./scripts/render.py --gpu 1 -s 25 -r 0 --render 
   ```
   There is a confirmation step after loading all the objects and before rendering. If there is no problem with the pose number, press enter to start rendering. Usually we need 150-200 poses for one room. You can modify the following parameters to increase or decrease number of poses. \
   Configurable parameters for rendering:
   - `--pos_per_obj`: Number of close-up poses for each object.
   - `--max_global_pos`: Max number of global poses.
   - `--global_density`: The radius interval of global poses. Smaller global_density -> more global views
    
