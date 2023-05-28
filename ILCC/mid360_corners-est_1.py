import argparse
import os
import numpy as np
import torch
import cv2
from typing import Tuple, Union
import open3d as o3d

import _pickle as cPickle

import time


def read_csv_point_cloud(csvfile: str) -> Tuple[np.array, np.array]:

    ## read csv
    csv_content = np.genfromtxt(csvfile, delimiter=",", skip_header=1)
    pcd, intensity = csv_content[:, :3], csv_content[:, 3]
    xyz = pcd

    ## return coordinates and intensity
    return xyz, intensity#rgb

def np_to_gpu(np_arr):
    np_arr = np.array(np_arr)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    torch_arr = torch.from_numpy(np_arr).float()
    torch_arr = torch_arr.to(device)
    return torch_arr

def rearrange_in_order(xyz,rgb):

    # project farther points first
    dist = torch.norm(xyz, dim=-1)
    mod_idx = torch.argsort(dist)
    mod_idx = torch.flip(mod_idx, dims=[0])

    ## rearrange xyz, rgb, dist
    mod_xyz = xyz.clone().detach()[mod_idx] ## xyz in ascending dist
    mod_rgb = rgb.clone().detach()[mod_idx] ## rgb in ascending dist
    mod_dist = dist.clone().detach()[mod_idx] ## dist in ascending dist

    return mod_xyz, mod_rgb, mod_dist

def get_pixel(xyz):
    # vertical angle
    theta = torch.unsqueeze(torch.atan2((torch.norm(xyz[:, :2], dim=-1)), xyz[:, 2] + 1e-6), 1)

    # horizontal angle
    phi = torch.atan2(xyz[:, 1:2], xyz[:, 0:1] + 1e-6)
    phi += np.pi

    # image coordinates ranged in [0, 1]
    sphere_cloud_arr = torch.cat([phi, theta], dim=-1)
    coord_arr = torch.stack([1.0 - sphere_cloud_arr[:, 0] / (np.pi * 2), sphere_cloud_arr[:, 1] / np.pi], dim=-1)

    ## normalized_coor
    ## [0,1] pixel_idx[:, 0] is x coordinate, pixel_idx[:, 1] is y coordinate
    pixel_idx = torch.zeros(coord_arr.shape, dtype=torch.int, device=coord_arr.device)
    pixel_idx[:, 0] = coord_arr[:,0] * (resolution[1] - 1) ## why -1, min will then be -1? ans: resolution is shape of array which larger than last array index by 1
    pixel_idx[:, 1] = coord_arr[:,1] * (resolution[0] - 1)

    ## flip dimension and change type
    pixel_idx = torch.flip(pixel_idx, [-1])
    pixel_idx = pixel_idx.long()
    pixel_idx = tuple(pixel_idx.t())

    return  pixel_idx

def xyz_from_pixel_and_depth(pixel_idx,depth):

    ## normalized_coor [0,1]
    coord_arr = np.zeros(pixel_idx.shape)
    coord_arr[:, 0] = pixel_idx[:,0] / (resolution[1] - 1)
    coord_arr[:, 1] = pixel_idx[:,1] / (resolution[0] - 1)

    # vertical angle
    theta = - (coord_arr[:,1] * np.pi - np.pi/2)

    # horizontal angle
    phi = (1.0 - coord_arr[:,0]) * (np.pi * 2)

    z = np.sin(theta) * depth
    phi -= np.pi
    x = np.cos(theta) * depth * np.cos(phi)
    y = np.cos(theta) * depth * np.sin(phi)

    xyz_res = np.stack([x,y,z], -1)

    return xyz_res





def make_padded_image(coord_idx,mod_rgb,resolution): ## from bonan

    with torch.no_grad():
        image = torch.zeros([resolution[0], resolution[1]], dtype=torch.float, device=mod_rgb.device)

        # color the image
        # pad by 1
        temp = torch.ones_like(coord_idx[0], device=mod_rgb.device)

        coord_idx1 = (torch.clamp(coord_idx[0] + temp, max=resolution[0] - 1),
                      torch.clamp(coord_idx[1] + temp, max=resolution[1] - 1))
        coord_idx2 = (torch.clamp(coord_idx[0] + temp, max=resolution[0] - 1),
                      coord_idx[1])
        coord_idx3 = (torch.clamp(coord_idx[0] + temp, max=resolution[0] - 1),
                      torch.clamp(coord_idx[1] - temp, min=0))
        coord_idx4 = (torch.clamp(coord_idx[0] - temp, min=0),
                      torch.clamp(coord_idx[1] + temp, max=resolution[1] - 1))
        coord_idx5 = (torch.clamp(coord_idx[0] - temp, min=0),
                      coord_idx[1])
        coord_idx6 = (torch.clamp(coord_idx[0] - temp, min=0),
                      torch.clamp(coord_idx[1] - temp, min=0))
        coord_idx7 = (coord_idx[0],
                      torch.clamp(coord_idx[1] + temp, max=resolution[1] - 1))
        coord_idx8 = (coord_idx[0],
                      torch.clamp(coord_idx[1] - temp, min=0))

        print(len(coord_idx))
        print(coord_idx8)

        image.index_put_(coord_idx8, mod_rgb, accumulate=False)

        image.index_put_(coord_idx7, mod_rgb, accumulate=False)
        image.index_put_(coord_idx6, mod_rgb, accumulate=False)
        image.index_put_(coord_idx5, mod_rgb, accumulate=False)
        image.index_put_(coord_idx4, mod_rgb, accumulate=False)
        image.index_put_(coord_idx3, mod_rgb, accumulate=False)
        image.index_put_(coord_idx2, mod_rgb, accumulate=False)
        image.index_put_(coord_idx1, mod_rgb, accumulate=False)
        image.index_put_(coord_idx, mod_rgb, accumulate=False)


    return np.array(image.cpu())

def make_pano_img_from_data(pixel_coor,data,resolution,kernel_size = 3):
    # pixel_coor = np.array(pixel_coor)
    # ys = pixel_coor[:,1].astype(int)
    # xs = pixel_coor[:,0].astype(int)

    ys = pixel_coor[0]
    xs = pixel_coor[1]


    image = torch.zeros(resolution)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    image = image.to(device)
    if len(data.shape) > 1: data = data[:,0]
    image[ys,xs] = data

    image = np.array(image.cpu())

    t = time.time()
    kernel = np.ones((kernel_size, kernel_size), np.uint8)

    # # image_high = image * (image > 125)
    # image_high = image
    # img_dilation = cv2.dilate(image_high, kernel, iterations=1)
    # img_high_diluate_pts = img_dilation * (image == 0)
    # image_final = image + img_high_diluate_pts
    # print('time.time() - t: ',time.time() - t)

    image_final = cv2.dilate(image, kernel, iterations=1)

    return image_final


def find_chessboard(image):
    gray = image

    # gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Detect the calibration board
    pattern_size = (8, 6)  # Number of inner corners on the board
    found, corners = cv2.findChessboardCorners(gray, pattern_size, None)

    print('found: ',found)
    # Draw the corners if the board is found
    if found:

        # img = image.copy()
        # # img = np.stack([img,img,img],-1)
        # for k in range(int(len(corners)/3)):
        #     (x,y) = corners.reshape(-1,2)[k].astype(int)
        #     cv2.circle(img, (x,y), 5, 255-k*3, -1)
        # cv2.imwrite(args.dataset_path + "/intensity_img/" + 'ordered_board_' + save_img_name, img)

        ## change order to fit ILCC
        corners = np.transpose(corners.reshape(pattern_size[1],pattern_size[0],2), (1, 0, 2)).reshape(-1,2)


        ## draw normal board
        cv2.drawChessboardCorners(image, pattern_size, corners, found)
        cv2.imwrite(args.dataset_path + "/intensity_img/" + 'board_' + save_img_name, image)
        # cv2.imshow('Calibration Board Detection', image)
        # cv2.waitKey(0)

        ## reverse order to fit ILCC LM_opt.py
        [H,W] = resolution
        corners_in_img_arr = corners
        # if np.linalg.norm(np.array(corners_in_img_arr[0]) - np.array([0, H])) > np.linalg.norm(
        #         np.array(corners_in_img_arr[-1]) - np.array([0, H])):
        #     print(csv_file + " is counted in reversed order")
        #     corners_in_img_arr = np.flipud(corners_in_img_arr)
        # elif corners_in_img_arr[0][1] < corners_in_img_arr[1][1]:
        #     print(csv_file + " is counted in reversed order, board orientation not good")
        #     corners_in_img_arr = np.flipud(corners_in_img_arr)
        # corners = corners_in_img_arr

         ## save corner in img to check order of corner
        img = image.copy()
        # img = np.stack([img,img,img],-1)
        for k in range(int(len(corners)/5*2)):
            (x,y) = corners.reshape(-1,2)[k].astype(int)
            cv2.circle(img, (x,y), 5, 255-k*3, -1)
        cv2.imwrite(args.dataset_path + "/intensity_img/" + 'new_ordered_board_' + save_img_name, img)


    else:
        print('Calibration board not found.')

    return found, corners

def save_xyz2pcd(xyz,base_dir):
    pcl = o3d.geometry.PointCloud()
    pcl.points = o3d.utility.Vector3dVector(xyz)
    o3d.io.write_point_cloud(base_dir + "/output/pcd_seg/" + str(file_index) + '_vis.pcd',pcl)

def save_xyz2pkl(xyz):

    print('xyz.shape: ',xyz.shape)
    corner_arr = xyz.reshape(-1,1,2)
    print('corner_arr.shape',corner_arr.shape)
    rot1 = np.identity(3)
    rot2 = np.identity(3)
    t1 = np.zeros(3)
    t2 = np.zeros(3)
    result = [rot1, t1, rot2, t2, corner_arr, "res.x", "os.path.relpath(marker_pkl[0])"]
    save_file_path = os.path.join(base_dir, "output/pcd_seg/") + str(i) + "_pcd_result.pkl"
    print('save_file_path: ',save_file_path)
    with open(os.path.abspath(save_file_path), 'wb') as file:
        file.truncate()
        cPickle.dump(result, file, protocol=2)

if __name__ == "__main__":

    ## set parameters
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_path', type=str, default="fisheyes_image_dataset")
    args = parser.parse_args()
    
    base_dir = args.dataset_path

    # resolution = (int(3040/3), int(6080/3))
    resolution = (3040, 6080)
    csv_base_path = os.path.join(args.dataset_path, "pcd")

    ## load path and filenames
    filelist__  = os.listdir(csv_base_path)
    filelist__.sort()
    print('filelist__: ', filelist__)

    ## loop
    for j in range(len(filelist__)):

        ## read file, get xyz, intensity
        # csv_file = filelist__[i]
        i = j + 1
        file_index = str(i)
        csv_file = file_index  + '.csv'
        save_img_name = csv_file[:-3] + 'jpg'

        print('i: ',i)
        print('j: ',j)

        print('csv_file: ',csv_file)
        print('save_img_name: ',save_img_name)

        xyz_np, rgb_np = read_csv_point_cloud(os.path.join(csv_base_path, csv_file))
        print('1')
        ## push to gpu memory
        xyz_gpu = np_to_gpu(xyz_np)
        rgb_gpu = np_to_gpu(rgb_np)
        print('1')

        ## rearrange in ascending dist
        xyz_gpu_ascending, rgb_gpu_ascending,dist_ascending = rearrange_in_order(xyz_gpu,rgb_gpu)

        ##  transform to pixel and depth
        pixel_coor_ascending = get_pixel(xyz_gpu_ascending)
        print('1')

        ## create intensity image
        # img_intensity = make_padded_image(pixel_coor_ascending,rgb_gpu_ascending,resolution)

        img_intensity = make_pano_img_from_data(pixel_coor_ascending,rgb_gpu_ascending,resolution)

        ## create intensity image
        img_depth = make_padded_image(pixel_coor_ascending,dist_ascending,resolution)

        # img_depth = make_pano_img_from_data(pixel_coor_ascending,rgb_gpu_ascending,resolution,kernel_size = 7)

        # cv2.imwrite(args.dataset_path + "/intensity_img/" + str(i + 1) + ".jpg", img_depth_intensity[:,:,1])
        cv2.imwrite(args.dataset_path + "/intensity_img/" + save_img_name, img_intensity)

        print('2')

        intensity_image = cv2.imread(args.dataset_path + "/intensity_img/" + save_img_name)

        ## detect board corners
        found, corners = find_chessboard(intensity_image)
        print('2')

        ## if found, get depth
        if found:
            corners = np.round(np.array(corners).reshape(-1,2),0).astype(int)
            corners_depth_ls =  img_depth[corners[:,1],corners[:,0]]
            print('corners.shape: ',corners.shape)

            print('corners_depth_ls')
            print(corners_depth_ls)
            # print('corners')
            # print(corners)

            temp_depth = 1
            for idx in range(len(corners_depth_ls)):
                if corners_depth_ls[idx] != 0:
                    temp_depth = corners_depth_ls[idx]
                else:
                    corners_depth_ls[idx] = temp_depth

            print('min(corners_depth_ls)',min(corners_depth_ls))


            ## transform phi,theta,depth to xyz
            xyz_res = xyz_from_pixel_and_depth(corners,corners_depth_ls)



            ## save corners in pkl
            save_xyz2pkl(xyz_res)

            ## save corners in xyz
            save_xyz2pcd(xyz_res, base_dir)

        else:
            print('No board detected for ',csv_file)


        # break




