import numpy as np
import cv2
import os
from load_helper import *
from render_configs import *

red = (0,0,255)
green = (0,255,0)
blue = (255,0,0)
class FloorPlan():
    def __init__(self, scene_idx):
        self.scene_idx = scene_idx
        self.names, self.bbox_mins, self.bbox_maxs = get_scene_bbox_meta(scene_idx)

        self.scene_min = np.min(self.bbox_mins, axis=0)
        self.scene_max = np.max(self.bbox_maxs, axis=0)
        print('scene_min:', self.scene_min)
        print('scene_max', self.scene_max)

        self.scale = 200
        self.margin = 100

        self.width = int((self.scene_max-self.scene_min)[0]*self.scale)+self.margin*2
        self.height = int((self.scene_max-self.scene_min)[1]*self.scale)+self.margin*2

        self.image = np.ones((self.height,self.width,3), np.uint8)

    def draw_coords(self):
        
        seg = 0.08
        x0, y0 = self.point_to_image([0,0,0])
        cv2.line(self.image, (0,y0), (self.width-1, y0), color=red, thickness=3)
        cv2.line(self.image, (x0,0), (x0, self.height-1), color=red, thickness=3)
        
        for i in range(int(np.floor(self.scene_min[0])), int(np.ceil(self.scene_max[0])+1)):
            cv2.line(self.image, self.point_to_image([i, -seg]), self.point_to_image([i, seg]), color=red, thickness=2)
        for i in range(int(np.floor(self.scene_min[1])), int(np.ceil(self.scene_max[1])+1)):
            cv2.line(self.image, self.point_to_image([-seg, i]), self.point_to_image([seg, i]), color=red, thickness=2)
        
        cv2.putText(self.image, 'x+', (self.width-80, y0-20), cv2.FONT_HERSHEY_SIMPLEX, 2, red, thickness=2)
        cv2.putText(self.image, 'y+', (x0+20, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, red, thickness=2)
    
    def draw_room_bbox(self):
        if self.scene_idx in ROOM_CONFIG.keys():
            for value in ROOM_CONFIG[self.scene_idx].values():
                scene_bbox = value['bbox']
                cv2.rectangle(self.image, self.point_to_image(scene_bbox[0]), self.point_to_image(scene_bbox[1]), color=blue, thickness=5)
    
    def draw_objects(self):
        for i in range(len(self.names)):
            x1, y1 = self.point_to_image(self.bbox_mins[i])
            x2, y2 = self.point_to_image(self.bbox_maxs[i])
            color = np.random.randint(0, 255, size=3)
            color = (int(color[0]), int(color[1]), int(color[2]))

            if self.names[i][:4] == 'Wall':
                cv2.rectangle(self.image, (x1, y1), (x2, y2), (255, 255, 0), -1)
            elif self.names[i][:5] == 'Floor':
                pass
            else:
                cv2.rectangle(self.image, (x1, y1), (x2, y2), color, 2)
                cv2.putText(self.image, self.names[i], (x2,y2), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color)
    
    def point_to_image(self, point_3d):
        """ Args: \\
                point_3d: raw float 3D point [x, y, z]
        """
        return int(point_3d[0]*self.scale - self.scene_min[0]*self.scale + self.margin), self.height-int(point_3d[1]*self.scale - self.scene_min[1]*self.scale + self.margin)
    
    def save(self, file_name, dst_dir):
        cv2.imwrite(os.path.join(dst_dir, file_name), self.image)
    
    def drawgroups_and_save(self, dst_dir):
        self.draw_objects()
        self.draw_coords()
        self.draw_room_bbox()
        self.save('floor_plan.jpg', dst_dir)



class FloorPlan_rot():
    def __init__(self, scene_idx):
        self.scene_idx = scene_idx
        self.names, self.bboxes = get_scene_rot_bbox_meta(scene_idx)

        
        self.bbox_mins = np.min(self.bboxes, axis=1)
        self.bbox_maxs = np.max(self.bboxes, axis=1)
        

        self.scene_min = np.min(self.bbox_mins, axis=0)
        self.scene_max = np.max(self.bbox_maxs, axis=0)
        print('scene_min:', self.scene_min)
        print('scene_max', self.scene_max)

        self.scale = 200
        self.margin = 100

        self.width = int((self.scene_max-self.scene_min)[0]*self.scale)+self.margin*2
        self.height = int((self.scene_max-self.scene_min)[1]*self.scale)+self.margin*2

        self.image = np.ones((self.height,self.width,3), np.uint8)

    def draw_coords(self):
        
        seg = 0.08
        x0, y0 = self.point_to_image([0,0,0])
        cv2.line(self.image, (0,y0), (self.width-1, y0), color=red, thickness=3)
        cv2.line(self.image, (x0,0), (x0, self.height-1), color=red, thickness=3)
        
        for i in range(int(np.floor(self.scene_min[0])), int(np.ceil(self.scene_max[0])+1)):
            cv2.line(self.image, self.point_to_image([i, -seg]), self.point_to_image([i, seg]), color=red, thickness=2)
        for i in range(int(np.floor(self.scene_min[1])), int(np.ceil(self.scene_max[1])+1)):
            cv2.line(self.image, self.point_to_image([-seg, i]), self.point_to_image([seg, i]), color=red, thickness=2)
        
        cv2.putText(self.image, 'x+', (self.width-80, y0-20), cv2.FONT_HERSHEY_SIMPLEX, 2, red, thickness=2)
        cv2.putText(self.image, 'y+', (x0+20, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, red, thickness=2)
    
    def draw_room_bbox(self):
        if self.scene_idx in ROOM_CONFIG.keys():
            for value in ROOM_CONFIG[self.scene_idx].values():
                scene_bbox = value['bbox']
                cv2.rectangle(self.image, self.point_to_image(scene_bbox[0]), self.point_to_image(scene_bbox[1]), color=blue, thickness=5)
    
    def draw_objects(self):
        for i in range(len(self.names)):

            x_list,y_list = [], []
            
            for j in range(8):
                temp = self.point_to_image(self.bboxes[i][j])
                x_list.append(temp[0])
                y_list.append(temp[1])
            
            if self.names[i] == 'chair.001':
                print(self.bboxes[i])
            
            x_min, y_min = self.point_to_image(self.bbox_mins[i])
            x_max, y_max = self.point_to_image(self.bbox_maxs[i])

            color = np.random.randint(0, 255, size=3)
            color = (int(color[0]), int(color[1]), int(color[2]))

            if self.names[i][:4] == 'Wall':
                cv2.rectangle(self.image, (x_min, y_min), (x_max, y_max), (255, 255, 0), -1)

                
            elif self.names[i][:5] == 'Floor':
                pass
            else:
                for j in range(3):
                    cv2.line(self.image, (x_list[j], y_list[j]), (x_list[j+1], y_list[j+1]), color, 2)
                    
                for j in range(3):
                    cv2.line(self.image, (x_list[j+4], y_list[j+4]), (x_list[j+5], y_list[j+5]), color, 2)
                cv2.putText(self.image, self.names[i], (x_list[0],y_list[0]), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color)
    
    def point_to_image(self, point_3d):
        """ Args: \\
                point_3d: raw float 3D point [x, y, z]
        """
        return int(point_3d[0]*self.scale - self.scene_min[0]*self.scale + self.margin), self.height-int(point_3d[1]*self.scale - self.scene_min[1]*self.scale + self.margin)
    
    def save(self, file_name, dst_dir):
        cv2.imwrite(os.path.join(dst_dir, file_name), self.image)
    
    def drawgroups_and_save(self, dst_dir):
        self.draw_objects()
        self.draw_coords()
        self.draw_room_bbox()
        self.save('floor_plan_rot.jpg', dst_dir)
