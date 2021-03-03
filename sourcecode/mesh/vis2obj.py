import cv2
import os
import numpy as np
import argparse
import random
import cv2

from sourcecode.configs import Options, make_config

FLOOR_CFG = {
    'color_maps': [
        [0, 0, 0],  # background
        [192, 192, 224],  # closet
        [192, 255, 255],  # batchroom/washroom
        [224, 255, 192],  # livingroom/kitchen/dining room
        [255, 224, 128],  # bedroom
        [255, 160, 96],  # hall
        [255, 224, 224],  # balcony
        [255, 60, 128],  # extra label for opening (door&window)
        [255, 255, 255],  # extra label for wall line
        [77, 77, 77] # ignore
    ],
    'avaliable_furniture': [
        ['None'],
        ['night_stand', 'dresser'],
        ['toilet', 'bathtub'],
        ['sofa'],
        ['bed'],
        ['dresser'],
        ['desk', 'table'],
        ['None'],
        ['None'],
        ['None']
    ]
}

AVALIABLE_ROOM_THRESHOLD = 500
ROOM_TYPE = ['background', 'closet', 'bashroom', 'livingroom', 'bedroom', 'hall', 'balcony']
FURNITURES = [
                'bathtub', 'bed', 'chair', 'desk', 'dresser', 
                'monitor', 'night_stand', 'sofa', 'table', 'toilet'
            ]

def to_index(mask_img):
    res_mask = np.array(np.zeros(mask_img.shape[:2]), dtype=np.uint8)
    for i in range(len(FLOOR_CFG['color_maps'])):
        res_mask[(mask_img == FLOOR_CFG['color_maps'][i]).all(2)] = i
    return res_mask

def get_furniture_info():
    print('loading furniture information...')
    furniture_info = dict()
    model_dir = './data/ModelNet10'
    for furniture in FURNITURES:
        furniture_info[furniture] = []
        sub_dir = os.path.join(model_dir, furniture)
        off_list = sorted(os.listdir(sub_dir))
        for off_path in off_list:
            if off_path[-3:] == 'off':
                max_point_x = -1000000
                min_point_x = 1000000
                max_point_y = -1000000
                min_point_y = 1000000
                max_point_z = -1000000
                min_point_z = 1000000
                with open(os.path.join(sub_dir, off_path)) as off_file:
                    data = off_file.readlines()
                    points = int(data[1].split(' ')[0])
                    for i in range(points):
                        point_data = data[2+i].split(' ')
                        point_x = float(point_data[0])
                        point_y = float(point_data[1])
                        point_z = float(point_data[2])
                        # assign values.
                        if point_x > max_point_x:
                            max_point_x = point_x
                        if point_x < min_point_x:
                            min_point_x = point_x
                        if point_y > max_point_y:
                            max_point_y = point_y
                        if point_y < min_point_y:
                            min_point_y = point_y
                        if point_z > max_point_z:
                            max_point_z = point_z
                        if point_z < min_point_z:
                            min_point_z = point_z

                    furniture_info[furniture].append(dict())
                    furniture_info[furniture][-1]['name'] = os.path.join(sub_dir, off_path)
                    furniture_info[furniture][-1]['width'] = max_point_x - min_point_x
                    furniture_info[furniture][-1]['length'] = max_point_y - min_point_y
                    furniture_info[furniture][-1]['height'] = max_point_z - min_point_z
                    furniture_info[furniture][-1]['x_offset'] = -min_point_x
                    furniture_info[furniture][-1]['y_offset'] = -min_point_y
                    furniture_info[furniture][-1]['z_offset'] = -min_point_z

    return furniture_info

def determine_room_type(label, furniture_info):
    walls = np.zeros((label.shape[0], label.shape[1])).astype(np.uint8)
    openings = np.zeros((label.shape[0], label.shape[1])).astype(np.uint8)
    rooms = np.ones((label.shape[0], label.shape[1])).astype(np.uint8)*255
    label_index = to_index(label)
    walls[(label == [255,255,255]).all(2)] = 255
    openings[(label == [255, 60, 128]).all(2)] = 255
    rooms[(label == [0,0,0]).all(2)] = 0
    rooms[walls==255] = 0
    rooms[openings==255] = 0

    points = []
    surfaces = []
    vertex_offset = 0

    # disconnect between regions.
    rooms = cv2.erode(rooms, np.ones((4,4)), 2)
    cv2.imwrite('rooms.png', rooms)
    num_obj, room_labels, stats, centroids = cv2.connectedComponentsWithStats(rooms)
    for i in range(0, num_obj):
        if stats[i][-1]>=AVALIABLE_ROOM_THRESHOLD:
            room_mask = np.zeros((label.shape[0], label.shape[1])).astype(np.uint8)
            room_mask[room_labels==i] = 1
            room_type = np.bincount((room_mask*label_index.copy()).astype(np.uint8).flatten())
            # avoid the background case.
            if len(room_type)>1 and len(room_type)<=7:
                room_type = np.argmax(room_type[1:])+1
                print(ROOM_TYPE[room_type])
                if FLOOR_CFG['avaliable_furniture'][room_type][0] != 'None':
                    
                    # contour detection for each room: get the area to place the furniture.
                    # only consider the external contour.
                    room_contours, _ = cv2.findContours(room_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    contour = cv2.approxPolyDP(room_contours[0], 1, True)
                    rect = order_points(contour.reshape(contour.shape[0], 2))
                    xs = [m[0] for m in rect]
                    ys = [m[1] for m in rect]
                    xs.sort()
                    ys.sort()
                    rect = np.array([[[xs[1], ys[1]]],
                                    [[xs[2], ys[1]]],
                                    [[xs[2], ys[2]]],
                                    [[xs[1], ys[2]]]], dtype = np.int64)
                    room_avaliable_info = []
                    rect_width = xs[2] - xs[1]
                    rect_length = ys[2] - ys[1]
                    
                    for furniture in FLOOR_CFG['avaliable_furniture'][room_type]:
                        room_avaliable_info += furniture_info[furniture]
                    
                    # random sample for a suitable furniture.
                    for k in range(100):
                        rand_index = random.randint(0, len(room_avaliable_info)-1)
                        seed = random.randint(0, 1)
                        
                        # horizontal.
                        if room_avaliable_info[rand_index]['width'] < rect_width and \
                            room_avaliable_info[rand_index]['length'] < rect_length and seed == 0:
                            off_file = open(room_avaliable_info[rand_index]['name'])
                            off_data = off_file.readlines()
                            num_pts = int(off_data[1].split(' ')[0])
                            num_surfaces = int(off_data[1].split(' ')[1])
                            x_spatial_offset = random.randint(0, int(rect_width - room_avaliable_info[rand_index]['width']))
                            y_spatial_offset = random.randint(0, int(rect_length - room_avaliable_info[rand_index]['length']))
                            for pt in range(num_pts):
                                point = off_data[pt+2].split(' ')
                                y = float(point[0])+room_avaliable_info[rand_index]['x_offset']+xs[1] + x_spatial_offset
                                x = float(point[1])+room_avaliable_info[rand_index]['y_offset']+ys[1] + y_spatial_offset
                                z = float(point[2])+room_avaliable_info[rand_index]['z_offset']+0.1
                                points.append([x,y,z])
                            for surf in range(num_surfaces):
                                surface = off_data[surf+2+num_pts].split(' ')
                                surfaces.append([int(surface[3])+vertex_offset,
                                                int(surface[2])+vertex_offset,
                                                int(surface[1])+vertex_offset,
                                                FLOOR_CFG['color_maps'][room_type][0]/255, 
                                                FLOOR_CFG['color_maps'][room_type][1]/255, 
                                                FLOOR_CFG['color_maps'][room_type][2]/255])
                            off_file.close()
                            vertex_offset += num_pts
                            break

                        # vertical.
                        elif room_avaliable_info[rand_index]['width'] < rect_length and \
                            room_avaliable_info[rand_index]['length'] < rect_width and seed == 1:
                            off_file = open(room_avaliable_info[rand_index]['name'])
                            off_data = off_file.readlines()
                            
                            num_pts = int(off_data[1].split(' ')[0])
                            num_surfaces = int(off_data[1].split(' ')[1])
                            x_spatial_offset = random.randint(0, int(rect_width - room_avaliable_info[rand_index]['length']))
                            y_spatial_offset = random.randint(0, int(rect_length - room_avaliable_info[rand_index]['width']))
                            for pt in range(num_pts):
                                point = off_data[pt+2].split(' ')
                                y = float(point[1])+room_avaliable_info[rand_index]['y_offset']+xs[1] + x_spatial_offset
                                x = float(point[0])+room_avaliable_info[rand_index]['x_offset']+ys[1] + y_spatial_offset
                                z = float(point[2])+room_avaliable_info[rand_index]['z_offset']+0.1
                                points.append([x,y,z])
                            for surf in range(num_surfaces):
                                surface = off_data[surf+2+num_pts].split(' ')
                                surfaces.append([int(surface[1])+vertex_offset,
                                                int(surface[2])+vertex_offset,
                                                int(surface[3])+vertex_offset,
                                                FLOOR_CFG['color_maps'][room_type][0]/255, 
                                                FLOOR_CFG['color_maps'][room_type][1]/255, 
                                                FLOOR_CFG['color_maps'][room_type][2]/255])
                            vertex_offset += num_pts
                            break

    return points, surfaces

# tool function: find max intersect rect.
# Reference: https://www.cnblogs.com/lzq116/p/11866642.html
def order_points(pts):
    rect = np.zeros((4, 2), dtype = np.int64)
    s = pts.sum(axis = 1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis = 1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect

def vis2mesh(cfg, furniture_info, wall_height=50, down_factor=1):
    img_folder = os.path.join(cfg.FOLDER, 'vistest')
    mesh_folder = os.path.join(cfg.FOLDER, 'mesh')
    os.makedirs(mesh_folder, exist_ok = True)
    img_list = os.listdir(img_folder)
    for img in img_list:
        vertex_offset = 0
        points = []
        surfaces = []

        off_filename = os.path.join(mesh_folder, img.replace('png', 'off'))
        img = cv2.imread(os.path.join(img_folder, img))
        img = cv2.resize(img, (1024//down_factor, 512//down_factor))
        label = img[:,512//down_factor:]

        furniture_points, furniture_surfaces = determine_room_type(label, furniture_info)

        texture = img[:, :512//down_factor]*0.5 + img[:, 512//down_factor:]*0.5
        h, w = (label.shape[0], label.shape[1])

        walls = np.zeros((label.shape[0], label.shape[1])).astype(np.uint8)
        walls[(label == [255,255,255]).all(2)] = 255

        bg = np.zeros((label.shape[0], label.shape[1])).astype(np.uint8)
        bg[(label != [0,0,0]).all(2)] = 1

        for i in range(label.shape[0]+1):
            for j in range(label.shape[1]+1):
                points.append([i, j, 0])
        vertex_offset = len(points)+len(furniture_points)
        
        # start assign points.
        for i in range(label.shape[0]):
            for j in range(label.shape[1]):
                if bg[i][j] != 0:
                    color = texture[i][j].astype(np.float32)/255
                    surfaces.append([i*(w+1)+j+1+len(furniture_points),
                                i*(w+1)+j+len(furniture_points),
                                (i+1)*(w+1)+j+len(furniture_points),
                                (i+1)*(w+1)+j+1+len(furniture_points),
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

        contours, _ = cv2.findContours(walls, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

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
            f.write(str(len(points)+len(furniture_points)) + ' ' + str(len(surfaces)+len(furniture_surfaces)) + ' 0\n')

            for i in range(len(furniture_points)):
                f.write('{} {} {}\n'.format(
                            str(int(furniture_points[i][0])),
                            str(int(furniture_points[i][1])),
                            str(int(furniture_points[i][2])),
                            ))

            for i in range(len(points)):
                f.write('{} {} {}\n'.format(
                            str(points[i][0]),
                            str(points[i][1]),
                            str(points[i][2]),
                            ))

            for i in range(len(furniture_surfaces)):
                f.write('{} {} {} {} {} {} {}\n'.format(
                            3,
                            str(int(furniture_surfaces[i][0])),
                            str(int(furniture_surfaces[i][1])),
                            str(int(furniture_surfaces[i][2])),
                            str(furniture_surfaces[i][3]),
                            str(furniture_surfaces[i][4]),
                            str(furniture_surfaces[i][5]),
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
    exp_cfg = make_config(os.path.join(folder_name, "exp.yaml"))
    furniture_info = None
    furniture_info = get_furniture_info()   
    vis2mesh(exp_cfg, furniture_info)