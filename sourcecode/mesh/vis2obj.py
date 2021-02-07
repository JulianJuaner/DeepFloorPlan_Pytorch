from sourcecode.dataset.dataset import FloorPlanDataset_cfg
import cv2
import os
import numpy as np
import argparse
import cv2

from sourcecode.configs import Options, make_config

CUBE_INDEX = [[2,1,0], [3,2,0],
              [4,5,6], [6,7,4],
              [5,1,2], [2,6,5],
              [0,1,5], [5,4,0],
              [7,6,3], [6,2,3],
              [0,4,7], [7,3,0]]
'''
-1 -1 -1 |  1 -1 -1 |  1  1 -1 | -1  1 -1
-1 -1  1 |  1 -1  1 |  1  1  1 | -1  1  1
'''

def vis2mesh(cfg, wall_height=50, down_factor=1):
    img_folder = os.path.join(cfg.FOLDER, 'vistest')
    mesh_folder = os.path.join(cfg.FOLDER, 'mesh')
    os.makedirs(mesh_folder, exist_ok = True)
    img_list = os.listdir(img_folder)
    for img in img_list:
        vertex_offset = 0
        surface_offset = 0

        points = []
        surfaces = []

        off_filename = os.path.join(mesh_folder, img.replace('png', 'off'))
        img = cv2.imread(os.path.join(img_folder, img))
        img = cv2.resize(img, (1024//down_factor, 512//down_factor))
        label = img[:,512//down_factor:]
        texture = img[:, :512//down_factor]*0.5 + img[:, 512//down_factor:]*0.5
        h, w = (label.shape[0], label.shape[1])

        walls = np.zeros((label.shape[0], label.shape[1])).astype(np.uint8)
        walls[(label == [255,255,255]).all(2)] = 255

        bg = np.zeros((label.shape[0], label.shape[1])).astype(np.uint8)
        bg[(label != [0,0,0]).all(2)] = 1

        for i in range(label.shape[0]+1):
            for j in range(label.shape[1]+1):
                points.append([i, j, 0])
        vertex_offset = len(points)
        
        # start assign points.
        for i in range(label.shape[0]):
            for j in range(label.shape[1]):
                if bg[i][j] != 0:
                    color = texture[i][j].astype(np.float32)/255
                    surfaces.append([i*(w+1)+j+1,i*(w+1)+j,(i+1)*(w+1)+j,(i+1)*(w+1)+j+1,
                                color[2], color[1], color[0]])


                if walls[i][j] == 255:
                    color = [255,128,128]
                    points.append([i, j, wall_height])
                    points.append([i+1, j, wall_height])
                    points.append([i+1, j+1, wall_height])
                    points.append([i, j+1, wall_height])
                    surfaces.append([vertex_offset+3,vertex_offset+0,vertex_offset+1,vertex_offset+2,
                                color[2], color[1], color[0]])
                    vertex_offset += 4
                '''
                offset_increment = 0
                if walls[i][j] == 1:
                    height = wall_height
                    offset_increment = 4
                else:
                    height = 1
                if bg[i][j] != 0:
                    color = texture[i][j].astype(np.float32)/255

                    i = i - 256//down_factor
                    j = j - 256//down_factor
                    if offset_increment == 8:
                        points.append([i, j, 0])
                        points.append([i+1, j, 0])
                        points.append([i+1, j+1, 0])
                        points.append([i, j+1, 0])
                        points.append([i, j, height])
                        points.append([i+1, j, height])
                        points.append([i+1, j+1, height])
                        points.append([i, j+1, height])
                        for k in range(len(CUBE_INDEX)):
                            surfaces.append([CUBE_INDEX[k][0]+vertex_offset,
                                            CUBE_INDEX[k][1]+vertex_offset,
                                            CUBE_INDEX[k][2]+vertex_offset,
                                            color[2], color[1], color[0]])
                    else:
                        points.append([i, j, 0])
                        points.append([i+1, j, 0])
                        points.append([i+1, j+1, 0])
                        points.append([i, j+1, 0])
                        for k in range(2):
                            surfaces.append([CUBE_INDEX[k][0]+vertex_offset,
                                            CUBE_INDEX[k][1]+vertex_offset,
                                            CUBE_INDEX[k][2]+vertex_offset,
                                            color[2], color[1], color[0]])
                    '''
                        

                    # vertex_offset += offset_increment
        contours, _ = cv2.findContours(walls, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
        #print(contours)
        for i in range(len(contours)):
            color = [255,128,128]
            contours[i] = cv2.approxPolyDP(contours[i], 0.5, True)
            for k in range(-1, len(contours[i])-1):
                points.append([contours[i][k][0][1], contours[i][k][0][0], 0])
                points.append([contours[i][k+1][0][1], contours[i][k+1][0][0], 0])
                points.append([contours[i][k+1][0][1], contours[i][k+1][0][0], wall_height])
                points.append([contours[i][k][0][1], contours[i][k][0][0], wall_height])
                surfaces.append([vertex_offset+3,vertex_offset+0,vertex_offset+1,vertex_offset+2,
                                color[2], color[1], color[0]])
                vertex_offset += 4

        print(off_filename, len(points), len(surfaces))
        # write to off.
        with open(off_filename, 'w+') as f:
            f.write('OFF\n')
            f.write(str(len(points)) + ' ' + str(len(surfaces)) + ' 0\n')
            for i in range(len(points)):
                f.write('{} {} {}\n'.format(
                            str(points[i][0]),
                            str(points[i][1]),
                            str(points[i][2]),
                            ))
                
            for i in range(len(surfaces)):
                f.write('{} {} {} {} {} {} {} {}\n'.format(
                            4,
                            str(int(surfaces[i][0])),
                            str(int(surfaces[i][1])),
                            str(int(surfaces[i][2])),
                            str(int(surfaces[i][3])),
                            
                            str(surfaces[i][4]),
                            str(surfaces[i][5]),
                            str(surfaces[i][6]),
                            ))

            f.close()
        cv2.imwrite('test.png', walls)


if "__main__" in __name__:
    # initialize exp configs.
    parser = argparse.ArgumentParser()
    OptionInit = Options(parser)
    parser = OptionInit.initialize(parser)
    opt = parser.parse_args()
    folder_name = opt.exp
    print(folder_name)
    exp_cfg = make_config(os.path.join(folder_name, "exp.yaml"))
    print(exp_cfg)
    vis2mesh(exp_cfg)