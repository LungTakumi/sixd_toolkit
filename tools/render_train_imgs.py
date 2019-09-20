# Author: Tomas Hodan (hodantom@cmp.felk.cvut.cz)
# Center for Machine Perception, Czech Technical University in Prague

# Renders RGB-D images of an object model.

import os
import sys
import math
import numpy as np
import cv2
# import scipy.misc

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pysixd import view_sampler, inout, misc, renderer

from params.dataset_params import get_dataset_params
from MeshPly import MeshPly
import matplotlib.pyplot as plt

def compute_projection(points_3D, transformation, internal_calibration):
    projections_2d = np.zeros((2, points_3D.shape[1]), dtype='float32')
    camera_projection = (internal_calibration.dot(transformation)).dot(points_3D)
    projections_2d[0, :] = camera_projection[0, :]/camera_projection[2, :]
    projections_2d[1, :] = camera_projection[1, :]/camera_projection[2, :]
    return projections_2d

def get_3D_corners(vertices):
    
    min_x = np.min(vertices[0,:])
    max_x = np.max(vertices[0,:])
    min_y = np.min(vertices[1,:])
    max_y = np.max(vertices[1,:])
    min_z = np.min(vertices[2,:])
    max_z = np.max(vertices[2,:])

    corners = np.array([[0,0,0],
                        [min_x, min_y, min_z],
                        [min_x, min_y, max_z],
                        [min_x, max_y, min_z],
                        [min_x, max_y, max_z],
                        [max_x, min_y, min_z],
                        [max_x, min_y, max_z],
                        [max_x, max_y, min_z],
                        [max_x, max_y, max_z]])

    corners = np.concatenate((np.transpose(corners), np.ones((1,9)) ), axis=0)
    return corners

edges_corners = [[1, 2], [1, 3], [1, 5], [2, 4], [2, 6], [3, 4], [3, 7], [4, 8], [5, 6], [5, 7], [6, 8], [7, 8]]

isVisualize = False

dataset = 'jefftest'
#dataset = 'hinterstoisser'
# dataset = 'tless'
# dataset = 'tudlight'
# dataset = 'rutgers'
# dataset = 'tejani'
# dataset = 'doumanoglou'
# dataset = 'toyotalight'

model_type = ''
cam_type = ''

if dataset == 'jefftest':
    # Range of object dist. in test images: 346.31 - 1499.84 mm - with extended GT
    # (there are only 3 occurrences under 400 mm)
    # Range of object dist. in test images: 600.90 - 1102.35 mm - with only original GT
    radii = [400] # Radii of the view sphere [mm]
    # radii = range(600, 1101, 100)
    # radii = range(400, 1501, 100)

    azimuth_range = (0, 2 * math.pi)
    elev_range = (0, 0.5 * math.pi)
elif dataset == 'hinterstoisser':
    # Range of object dist. in test images: 346.31 - 1499.84 mm - with extended GT
    # (there are only 3 occurrences under 400 mm)
    # Range of object dist. in test images: 600.90 - 1102.35 mm - with only original GT
    radii = [400] # Radii of the view sphere [mm]
    # radii = range(600, 1101, 100)
    # radii = range(400, 1501, 100)

    azimuth_range = (0, 2 * math.pi)
    elev_range = (0, 0.5 * math.pi)

elif dataset == 'tless':
    # Range of object distances in test images: 649.89 - 940.04 mm
    radii = [650] # Radii of the view sphere [mm]
    # radii = range(500, 1101, 100) # [mm]

    azimuth_range = (0, 2 * math.pi)
    elev_range = (-0.5 * math.pi, 0.5 * math.pi)

    model_type = 'reconst'
    cam_type = 'primesense'

elif dataset == 'tudlight':
    # Range of object distances in test images: 851.29 - 2016.14 mm
    radii = [850] # Radii of the view sphere [mm]
    # radii = range(500, 1101, 100) # [mm]

    azimuth_range = (0, 2 * math.pi)
    elev_range = (-0.4363, 0.5 * math.pi) # (-25, 90) [deg]

elif dataset == 'rutgers':
    # Range of object distances in test images: 594.41 - 739.12 mm
    radii = [590] # Radii of the view sphere [mm]
    # radii = range(500, 1101, 100) # [mm]

    azimuth_range = (0, 2 * math.pi)
    elev_range = (-0.5 * math.pi, 0.5 * math.pi)

elif dataset == 'tejani':
    # Range of object dist. in test images: 509.12 - 1120.41 mm
    radii = [500] # Radii of the view sphere [mm]
    # radii = range(500, 1101, 100)

    azimuth_range = (0, 2 * math.pi)
    elev_range = (0, 0.5 * math.pi)

elif dataset == 'doumanoglou':
    # Range of object dist. in test images: 454.56 - 1076.29 mm
    radii = [450] # Radii of the view sphere [mm]
    # radii = range(500, 1101, 100)

    azimuth_range = (0, 2 * math.pi)
    elev_range = (-1.0297, 0.5 * math.pi) # (-59, 90) [deg]

par = get_dataset_params(dataset, model_type=model_type, cam_type=cam_type)

# Objects to render
obj_ids = range(1, par['obj_count'] + 1)

# Minimum required number of views on the whole view sphere. The final number of
# views depends on the sampling method.
min_n_views = 1000  # 1000

clip_near = 10 # [mm]
clip_far = 10000 # [mm]
ambient_weight = 0.8 # Weight of ambient light [0, 1]
shading = 'phong' # 'flat', 'phong'

# Super-sampling anti-aliasing (SSAA)
# https://github.com/vispy/vispy/wiki/Tech.-Antialiasing
# The RGB image is rendered at ssaa_fact times higher resolution and then
# down-sampled to the required resolution.
ssaa_fact = 4

# Output path masks
out_rgb_mpath = '../output/'+ dataset +'/{:02d}/JPEGImages/{:04d}.jpg'
out_mask_mpath = '../output/'+ dataset +'/{:02d}/mask/{:04d}.png'
out_obj_info_path = '../output/'+ dataset +'/{:02d}/info.yml'
out_obj_gt_path = '../output/'+ dataset +'/{:02d}/gt.yml'
out_views_vis_mpath = '../output/'+ dataset +'/views_radius={}.ply'
out_train_text_path = '../output/' + dataset + '/{:02d}/train.txt'
out_labels_path = '../output/'+ dataset +'/{:02d}/labels/{:04d}.txt'

# Prepare output folder
# misc.ensure_dir(os.path.dirname(out_obj_info_path))

# Image size and K for SSAA
im_size_rgb = [int(round(x * float(ssaa_fact))) for x in par['cam']['im_size']]
K_rgb = par['cam']['K'] * ssaa_fact

for obj_id in obj_ids:
    # Prepare folders
    misc.ensure_dir(os.path.dirname(out_rgb_mpath.format(obj_id, 0)))
    misc.ensure_dir(os.path.dirname(out_labels_path.format(obj_id, 0)))
    misc.ensure_dir(os.path.dirname(out_mask_mpath.format(obj_id, 0)))

    # Load model
    model_path = par['model_mpath'].format(obj_id)
    model = inout.load_ply(model_path)

    mesh          = MeshPly(model_path)
    vertices      = np.c_[np.array(mesh.vertices), np.ones((len(mesh.vertices), 1))].transpose()
    corners3D     = get_3D_corners(vertices)

    # Load model texture
    if par['model_texture_mpath']:
        model_texture_path = par['model_texture_mpath'].format(obj_id)
        model_texture = inout.load_im(model_texture_path)
    else:
        model_texture = None

    obj_info = {}
    obj_gt = {}
    im_id = 0
    trainText = ""
    for radius in radii:
        # Sample views
        views, views_level = view_sampler.sample_views(min_n_views, radius,
                                                       azimuth_range, elev_range)
        print('Sampled views: ' + str(len(views)))
        view_sampler.save_vis(out_views_vis_mpath.format(str(radius)),
                              views, views_level)

        # Render the object model from all the views
        for view_id, view in enumerate(views):
            if view_id % 10 == 0:
                print('obj,radius,view: ' + str(obj_id) +
                      ',' + str(radius) + ',' + str(view_id))

            # Render RGB image
            rgb = renderer.render(model, im_size_rgb, K_rgb, view['R'], view['t'],
                                  clip_near, clip_far, texture=model_texture,
                                  ambient_weight=ambient_weight, shading=shading,
                                  mode='rgb')

            # The OpenCV function was used for rendering of the training images
            # provided for the SIXD Challenge 2017.
            rgb = cv2.resize(rgb, par['cam']['im_size'], interpolation=cv2.INTER_AREA)
            #rgb = scipy.misc.imresize(rgb, par['cam']['im_size'][::-1], 'bicubic')

            # Save the rendered images
            inout.save_im(out_rgb_mpath.format(obj_id, im_id), rgb)

            mask = renderer.render(model, im_size_rgb, K_rgb, view['R'], view['t'],
                                  clip_near, clip_far, texture=None,
                                  ambient_weight=ambient_weight, shading='mask',
                                  mode='rgb')

            # The OpenCV function was used for rendering of the training images
            # provided for the SIXD Challenge 2017.
            mask = cv2.resize(mask, par['cam']['im_size'], interpolation=cv2.INTER_AREA)
            #rgb = scipy.misc.imresize(rgb, par['cam']['im_size'][::-1], 'bicubic')

            # Save the rendered images
            inout.save_im(out_mask_mpath.format(obj_id, im_id), mask)

            trainText += out_rgb_mpath.format(obj_id, im_id).replace('../output/', '') + "\n"

            Rt_gt        = np.concatenate((view['R'], view['t']), axis=1)
            proj_2D = np.transpose(compute_projection(corners3D, Rt_gt, par['cam']['K']))

            min_x = np.min(proj_2D[:,0])
            max_x = np.max(proj_2D[:,0])
            min_y = np.min(proj_2D[:,1])
            max_y = np.max(proj_2D[:,1])

            rangeX = max_x - min_x
            rangeY = max_y - min_y

            #if isVisualize:
            #    plt.xlim((0, par['cam']['im_size'][0]))
            #    plt.ylim((0, par['cam']['im_size'][1]))
            #    plt.imshow(rgb)

            #    plt.plot(proj_2D[0][0], proj_2D[0][1], 'co')

            #    for edge in edges_corners:
            #        plt.plot(proj_2D[edge, 0], proj_2D[edge, 1], color='g', linewidth=1)

            #    plt.plot([min_x, max_x], [min_y, min_y], color='b', linewidth=1)
            #    plt.plot([max_x, max_x], [min_y, max_y], color='b', linewidth=1)
            #    plt.plot([max_x, min_x], [max_y, max_y], color='b', linewidth=1)
            #    plt.plot([min_x, min_x], [max_y, min_y], color='b', linewidth=1)

            #    plt.gca().invert_yaxis()
            #    plt.show()

            label = []
            label.append(obj_id)

            proj_2D_n = proj_2D
            proj_2D_n[:, 0] = proj_2D[:, 0] / par['cam']['im_size'][0]
            proj_2D_n[:, 1] = proj_2D[:, 1] / par['cam']['im_size'][1]
            for pt in proj_2D_n:
                label.append(pt[0])
                label.append(pt[1])

            

            label.append(rangeX / par['cam']['im_size'][0])
            label.append(rangeY / par['cam']['im_size'][1])

            labelTxt = ""
            for data in label:
                labelTxt += str(data) + ' '
            with open(out_labels_path.format(obj_id, im_id), 'w') as f:
                f.write(labelTxt)

            

            im_id += 1

    # Save metadata
    with open(out_train_text_path.format(obj_id), 'w') as f:
        f.write(trainText)
