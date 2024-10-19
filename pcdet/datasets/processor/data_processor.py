from functools import partial

import cv2
import numpy as np
from skimage import transform
import torch
import torchvision
from ...utils import box_utils, common_utils

tv = None
try:
    import cumm.tensorview as tv
except:
    pass


class VoxelGeneratorWrapper():
    def __init__(self, vsize_xyz, coors_range_xyz, num_point_features, max_num_points_per_voxel, max_num_voxels):
        try:
            from spconv.utils import VoxelGeneratorV2 as VoxelGenerator
            self.spconv_ver = 1
        except:
            try:
                from spconv.utils import VoxelGenerator
                self.spconv_ver = 1
            except:
                from spconv.utils import Point2VoxelCPU3d as VoxelGenerator
                self.spconv_ver = 2

        if self.spconv_ver == 1:
            self._voxel_generator = VoxelGenerator(
                voxel_size=vsize_xyz,
                point_cloud_range=coors_range_xyz,
                max_num_points=max_num_points_per_voxel,
                max_voxels=max_num_voxels
            )
        else:
            self._voxel_generator = VoxelGenerator(
                vsize_xyz=vsize_xyz,
                coors_range_xyz=coors_range_xyz,
                num_point_features=num_point_features,
                max_num_points_per_voxel=max_num_points_per_voxel,
                max_num_voxels=max_num_voxels
            )

    def generate(self, points):
        if self.spconv_ver == 1:
            voxel_output = self._voxel_generator.generate(points)
            if isinstance(voxel_output, dict):
                voxels, coordinates, num_points = \
                    voxel_output['voxels'], voxel_output['coordinates'], voxel_output['num_points_per_voxel']
            else:
                voxels, coordinates, num_points = voxel_output
        else:
            assert tv is not None, f"Unexpected error, library: 'cumm' wasn't imported properly."
            voxel_output = self._voxel_generator.point_to_voxel(tv.from_numpy(points))
            tv_voxels, tv_coordinates, tv_num_points = voxel_output
            # make copy with numpy(), since numpy_view() will disappear as soon as the generator is deleted
            voxels = tv_voxels.numpy()
            coordinates = tv_coordinates.numpy()
            num_points = tv_num_points.numpy()
        return voxels, coordinates, num_points


class DataProcessor(object):
    def __init__(self, processor_configs, point_cloud_range, training, num_point_features):
        self.point_cloud_range = point_cloud_range
        self.training = training
        self.num_point_features = num_point_features
        self.mode = 'train' if training else 'test'
        self.grid_size = self.voxel_size = None
        self.data_processor_queue = []

        self.voxel_generator = None

        for cur_cfg in processor_configs:
            cur_processor = getattr(self, cur_cfg.NAME)(config=cur_cfg)
            self.data_processor_queue.append(cur_processor)

    def mask_points_and_boxes_outside_range(self, data_dict=None, config=None):
        if data_dict is None:
            return partial(self.mask_points_and_boxes_outside_range, config=config)

        if data_dict.get('points', None) is not None:
            mask = common_utils.mask_points_by_range(data_dict['points'], self.point_cloud_range)
            data_dict['points'] = data_dict['points'][mask]

        if data_dict.get('gt_boxes', None) is not None and config.REMOVE_OUTSIDE_BOXES and self.training:
            mask = box_utils.mask_boxes_outside_range_numpy(
                data_dict['gt_boxes'], self.point_cloud_range, min_num_corners=config.get('min_num_corners', 1), 
                use_center_to_filter=config.get('USE_CENTER_TO_FILTER', True)
            )
            data_dict['gt_boxes'] = data_dict['gt_boxes'][mask]
        return data_dict

    def shuffle_points(self, data_dict=None, config=None):
        if data_dict is None:
            return partial(self.shuffle_points, config=config)

        if config.SHUFFLE_ENABLED[self.mode]:
            points = data_dict['points']
            shuffle_idx = np.random.permutation(points.shape[0])
            points = points[shuffle_idx]
            data_dict['points'] = points

        return data_dict

    def transform_points_to_voxels_placeholder(self, data_dict=None, config=None):
        # just calculate grid size
        if data_dict is None:
            grid_size = (self.point_cloud_range[3:6] - self.point_cloud_range[0:3]) / np.array(config.VOXEL_SIZE)
            self.grid_size = np.round(grid_size).astype(np.int64)
            self.voxel_size = config.VOXEL_SIZE
            return partial(self.transform_points_to_voxels_placeholder, config=config)
        
        return data_dict

    def double_flip(self, points):
        # y flip
        points_yflip = points.copy()
        points_yflip[:, 1] = -points_yflip[:, 1]

        # x flip
        points_xflip = points.copy()
        points_xflip[:, 0] = -points_xflip[:, 0]

        # x y flip
        points_xyflip = points.copy()
        points_xyflip[:, 0] = -points_xyflip[:, 0]
        points_xyflip[:, 1] = -points_xyflip[:, 1]

        return points_yflip, points_xflip, points_xyflip

    def transform_points_to_voxels(self, data_dict=None, config=None):
        if data_dict is None:
            grid_size = (self.point_cloud_range[3:6] - self.point_cloud_range[0:3]) / np.array(config.VOXEL_SIZE)
            self.grid_size = np.round(grid_size).astype(np.int64)
            self.voxel_size = config.VOXEL_SIZE
            # just bind the config, we will create the VoxelGeneratorWrapper later,
            # to avoid pickling issues in multiprocess spawn
            return partial(self.transform_points_to_voxels, config=config)

        if self.voxel_generator is None:
            self.voxel_generator = VoxelGeneratorWrapper(
                vsize_xyz=config.VOXEL_SIZE,
                coors_range_xyz=self.point_cloud_range,
                num_point_features=self.num_point_features,
                max_num_points_per_voxel=config.MAX_POINTS_PER_VOXEL,
                max_num_voxels=config.MAX_NUMBER_OF_VOXELS[self.mode],
            )
        if config.USE_EMPTY_VOXEL:
            self.empty_voxel_generator = VoxelGeneratorWrapper(
                vsize_xyz=config.VOXEL_SIZE,
                coors_range_xyz=self.point_cloud_range,
                num_point_features=8,
                max_num_points_per_voxel=config.MAX_POINTS_PER_VOXEL,
                max_num_voxels=config.MAX_NUMBER_OF_VOXELS[self.mode],
            )
        points = data_dict['points']
        voxel_output = self.voxel_generator.generate(points)
        voxels, coordinates, num_points = voxel_output

        if config.USE_EMPTY_VOXEL:
            device='cuda'
            #=====构建三维空间======#
            xs = torch.arange(0, 320, 1, dtype=torch.float).view(1, 320).expand(320, 320).to(device)
            ys = torch.arange(0, 320, 1, dtype=torch.float).view(320, 1).expand(320, 320).to(device)
            hs = torch.zeros((320, 320)).to(device)
            mask = torch.zeros((320, 320)).to(device)
            frustum = torch.stack((hs, ys, xs,mask), -1)
            # frustum = frustum.reshape(-1, 3)
            del xs,ys,hs,mask
            #=====过滤有点的体素======#
            # mask = [] 下面的循环要不要
            for index in range(len(coordinates)):
                h,y,x = coordinates[index]
                frustum[y,x,3] = 1
            frustum = frustum.reshape(-1, 4)
            empty_pillars = frustum[frustum[:,3]==0]
            del frustum
            #=====将体素转化为点集3D======#
            empty_pillars = empty_pillars[:,:3]
            length = len(empty_pillars)
            pts = np.concatenate([empty_pillars,empty_pillars,empty_pillars],axis=0)
            del empty_pillars
            pts[:length,0] = -3
            pts[length:2*length,0] = -1
            pts[2*length:,0] = 1
            pts[:,1] = pts[:,1]*0.16-25.6
            pts[:,2] = pts[:,2]*0.16
            pts = pts[:,(2,1,0)]
            #=========投影点过滤=========#
            Calib = data_dict['calib']
            pts_img, pts_depth = Calib.lidar_to_img(pts)
            mask_x = np.logical_and(pts_img[:, 0] >= 0, pts_img[:, 0] < 1936)
            mask_y = np.logical_and(pts_img[:, 1] >= 0, pts_img[:, 1] < 1216)
            mask = np.logical_and(mask_x, mask_y)
            pts = pts[mask]
            pts_img = pts_img[mask]
            del mask,mask_x,mask_y
            #=========获取图像特征========#
            image = data_dict['images']
            image_features = image[pts_img[:,1].astype(int),pts_img[:,0].astype(int)]
            pts_img[:,0] = pts_img[:,0]/1936
            pts_img[:,1] = pts_img[:,1]/1216
            empty_pillars_features = np.concatenate([pts,pts_img,image_features],axis=1)
            del pts,pts_img,pts_depth,image_features
            empty_voxel_output = self.empty_voxel_generator.generate(empty_pillars_features)
            voxels_empty, coordinates_empty, num_points_empty = empty_voxel_output

        if config.USE_EMPTY_VOXEL:
            #=====构建三维空间======#
            xs = np.arange(0, 320, 1, dtype=float).reshape(1, 320).repeat(320, 0)
            ys = np.arange(0, 320, 1, dtype=float).reshape(320, 1).repeat(320, 1)
            hs = np.zeros((320, 320))
            mask = np.zeros((320, 320))
            frustum = np.stack((hs, ys, xs,mask), -1)
            del xs,ys,hs,mask
            # frustum = frustum.reshape(-1, 3)
            #=====过滤有点的体素======#
            # mask = []
            for index in range(len(coordinates)):
                h,y,x = coordinates[index]
                frustum[y,x,3] = 1
            frustum = frustum.reshape(-1, 4)
            empty_pillars = frustum[frustum[:,3]==0]
            del frustum
            #=====将体素转化为点集3D======#
            empty_pillars = empty_pillars[:,:3]
            length = len(empty_pillars)
            pts = np.concatenate([empty_pillars,empty_pillars,empty_pillars],axis=0)
            del empty_pillars
            pts[:length,0] = -3
            pts[length:2*length,0] = -1
            pts[2*length:,0] = 1
            pts[:,1] = pts[:,1]*0.16-25.6
            pts[:,2] = pts[:,2]*0.16
            pts = pts[:,(2,1,0)]
            #=========投影点过滤=========#
            Calib = data_dict['calib']
            pts_img, pts_depth = Calib.lidar_to_img(pts)
            mask_x = np.logical_and(pts_img[:, 0] >= 0, pts_img[:, 0] < 1936)
            mask_y = np.logical_and(pts_img[:, 1] >= 0, pts_img[:, 1] < 1216)
            mask = np.logical_and(mask_x, mask_y)
            pts = pts[mask]
            pts_img = pts_img[mask]
            del mask,mask_x,mask_y
            # =========获取图像特征========#
            image = data_dict['images']
            image_features = image[pts_img[:,1].astype(int),pts_img[:,0].astype(int)]
            pts_img[:,0] = pts_img[:,0]/1936
            pts_img[:,1] = pts_img[:,1]/1216
            empty_pillars_features = np.concatenate([pts,pts_img,image_features],axis=1)
            del pts,pts_img,pts_depth,image_features
            empty_voxel_output = self.empty_voxel_generator.generate(empty_pillars_features)
            voxels_empty, coordinates_empty, num_points_empty = empty_voxel_output

        if not data_dict['use_lead_xyz']:
            voxels = voxels[..., 3:]  # remove xyz in voxels(N, 3)

        if config.get('DOUBLE_FLIP', False):
            voxels_list, voxel_coords_list, voxel_num_points_list = [voxels], [coordinates], [num_points]
            points_yflip, points_xflip, points_xyflip = self.double_flip(points)
            points_list = [points_yflip, points_xflip, points_xyflip]
            keys = ['yflip', 'xflip', 'xyflip']
            for i, key in enumerate(keys):
                voxel_output = self.voxel_generator.generate(points_list[i])
                voxels, coordinates, num_points = voxel_output

                if not data_dict['use_lead_xyz']:
                    voxels = voxels[..., 3:]
                voxels_list.append(voxels)
                voxel_coords_list.append(coordinates)
                voxel_num_points_list.append(num_points)

            data_dict['voxels'] = voxels_list
            data_dict['voxel_coords'] = voxel_coords_list
            data_dict['voxel_num_points'] = voxel_num_points_list
        else:
            data_dict['voxels'] = voxels
            data_dict['voxel_coords'] = coordinates
            data_dict['voxel_num_points'] = num_points
            if config.USE_EMPTY_VOXEL:
                data_dict['voxels_empty'] = voxels_empty
                data_dict['coordinates_empty'] = coordinates_empty
                data_dict['num_points_empty'] = num_points_empty
                del data_dict['images']
            # run demo.py need #######
            grid_size = config['VOXEL_SIZE'][0] # 默认grid是正方形，取出边长
            voxel_label_length = int(51.2//grid_size)
            voxel_label = np.zeros((voxel_label_length,voxel_label_length))
            instance_label = np.zeros((voxel_label_length, voxel_label_length,3))
            instance_label[:,:,:] = 255
            vis_coor = np.zeros((voxel_label_length,voxel_label_length))
            for index in coordinates:
                _,x,y = index
                vis_coor[x,y] = 255
            # cv2.imshow('coor',vis_coor)
            # cv2.waitKey()
            # cv2.destroyWindow()
            coor_3d = coordinates.copy()
            coor_3d = coor_3d.astype(float)
            coor_3d[:,1] = coor_3d[:,1] * 0.16 - 25.6
            coor_3d[:,2] = coor_3d[:,2] * 0.16

            coor_3d = coor_3d[:,(2,1,0)]
            # index = data_dict['frame_id']
            gt_boxes = data_dict['gt_boxes']
            gt_classes = gt_boxes[:,-1]
            gt_boxes = gt_boxes[:,:-1]
            gt_boxes_corners = box_utils.boxes_to_corners_3d(gt_boxes)
            colors = {0:[255,255,0],1:[0,255,0],2:[255,0,255]}
            for i in range(len(gt_classes)):
                gt_class = gt_classes[i]
                gt_box_corners = gt_boxes_corners[i]
                gt_box_corners_low = gt_box_corners[0,2]
                gt_box_corners_height = gt_box_corners[-1,2]
                gt_box_corners_mid = (gt_box_corners_height+gt_box_corners_low)/2.0
                coor_3d[:, 2] = gt_box_corners_mid
                coor_bool = box_utils.in_hull(coor_3d,gt_box_corners)
                # voxel_label[coordinates[coor_bool][:,1],coordinates[coor_bool][:,2]] = gt_class
                # voxel_label[coordinates[coor_bool][:, 1], coordinates[coor_bool][:, 2]] = 1
                # instance_label[coordinates[coor_bool][:, 1], coordinates[coor_bool][:, 2]] = colors[int(gt_class-1)]
                for pix_index in range(len(coordinates[coor_bool][:, 2])):
                    cv2.circle(instance_label, (coordinates[coor_bool][pix_index, 2],coordinates[coor_bool][pix_index, 1]), 1, colors[int(gt_class-1)], 2)

                # instance_label[coordinates[coor_bool][:, 1], coordinates[coor_bool][:, 2],int(gt_class-1)] = 255
            data_dict['voxel_label'] = np.array(voxel_label,dtype=int)
            # data_dict['instance_label'] = instance_label
            save_dir = '/home/rpf/Pillarseg/OpenPCDet/data/view_of_delft/radar/instance_label_write/'
            frame_id = data_dict['frame_id']
            cv2.imwrite(save_dir+frame_id+'.png',instance_label)
            # cv2.imshow('voxel_label',instance_label)
            # cv2.waitKey()
            # cv2.destroyWindow()

            # for gt in gt_boxes:
            #     gt_class = gt[-1]
            #     gt_box = gt[:-1]
            #     gt_box_corner = box_utils.boxes_to_corners_3d(gt_box)



        return data_dict

    def sample_points(self, data_dict=None, config=None):
        if data_dict is None:
            return partial(self.sample_points, config=config)

        num_points = config.NUM_POINTS[self.mode]
        if num_points == -1:
            return data_dict

        points = data_dict['points']
        if num_points < len(points):
            pts_depth = np.linalg.norm(points[:, 0:3], axis=1)
            pts_near_flag = pts_depth < 40.0
            far_idxs_choice = np.where(pts_near_flag == 0)[0]
            near_idxs = np.where(pts_near_flag == 1)[0]
            choice = []
            if num_points > len(far_idxs_choice):
                near_idxs_choice = np.random.choice(near_idxs, num_points - len(far_idxs_choice), replace=False)
                choice = np.concatenate((near_idxs_choice, far_idxs_choice), axis=0) \
                    if len(far_idxs_choice) > 0 else near_idxs_choice
            else: 
                choice = np.arange(0, len(points), dtype=np.int32)
                choice = np.random.choice(choice, num_points, replace=False)
            np.random.shuffle(choice)
        else:
            choice = np.arange(0, len(points), dtype=np.int32)
            if num_points > len(points):
                extra_choice = np.random.choice(choice, num_points - len(points), replace=False)
                choice = np.concatenate((choice, extra_choice), axis=0)
            np.random.shuffle(choice)
        data_dict['points'] = points[choice]
        return data_dict

    def calculate_grid_size(self, data_dict=None, config=None):
        if data_dict is None:
            grid_size = (self.point_cloud_range[3:6] - self.point_cloud_range[0:3]) / np.array(config.VOXEL_SIZE)
            self.grid_size = np.round(grid_size).astype(np.int64)
            self.voxel_size = config.VOXEL_SIZE
            return partial(self.calculate_grid_size, config=config)
        return data_dict

    def downsample_depth_map(self, data_dict=None, config=None):
        if data_dict is None:
            self.depth_downsample_factor = config.DOWNSAMPLE_FACTOR
            return partial(self.downsample_depth_map, config=config)

        data_dict['depth_maps'] = transform.downscale_local_mean(
            image=data_dict['depth_maps'],
            factors=(self.depth_downsample_factor, self.depth_downsample_factor)
        )
        return data_dict
    
    def image_normalize(self, data_dict=None, config=None):
        if data_dict is None:
            return partial(self.image_normalize, config=config)
        mean = config.mean
        std = config.std
        compose = torchvision.transforms.Compose(
            [
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(mean=mean, std=std),
            ]
        )
        data_dict["camera_imgs"] = [compose(img) for img in data_dict["camera_imgs"]]
        return data_dict
    
    def image_calibrate(self,data_dict=None, config=None):
        if data_dict is None:
            return partial(self.image_calibrate, config=config)
        img_process_infos = data_dict['img_process_infos']
        transforms = []
        for img_process_info in img_process_infos:
            resize, crop, flip, rotate = img_process_info

            rotation = torch.eye(2)
            translation = torch.zeros(2)
            # post-homography transformation
            rotation *= resize
            translation -= torch.Tensor(crop[:2])
            if flip:
                A = torch.Tensor([[-1, 0], [0, 1]])
                b = torch.Tensor([crop[2] - crop[0], 0])
                rotation = A.matmul(rotation)
                translation = A.matmul(translation) + b
            theta = rotate / 180 * np.pi
            A = torch.Tensor(
                [
                    [np.cos(theta), np.sin(theta)],
                    [-np.sin(theta), np.cos(theta)],
                ]
            )
            b = torch.Tensor([crop[2] - crop[0], crop[3] - crop[1]]) / 2
            b = A.matmul(-b) + b
            rotation = A.matmul(rotation)
            translation = A.matmul(translation) + b
            transform = torch.eye(4)
            transform[:2, :2] = rotation
            transform[:2, 3] = translation
            transforms.append(transform.numpy())
        data_dict["img_aug_matrix"] = transforms
        return data_dict

    def forward(self, data_dict):
        """
        Args:
            data_dict:
                points: (N, 3 + C_in)
                gt_boxes: optional, (N, 7 + C) [x, y, z, dx, dy, dz, heading, ...]
                gt_names: optional, (N), string
                ...

        Returns:
        """

        for cur_processor in self.data_processor_queue:
            data_dict = cur_processor(data_dict=data_dict)

        return data_dict
