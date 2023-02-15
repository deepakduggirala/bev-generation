from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import scipy
from nuscenes.nuscenes import NuScenes
from sklearn.neighbors import KNeighborsClassifier

import nusc_utils
import utils


def interpolate_static_classes_bev(bev_seg_map):
    nc, H, W = bev_seg_map.shape
    xx, yy = np.meshgrid(np.arange(H), np.arange(W))
    xi = np.hstack((yy.reshape(-1, 1), xx.reshape(-1, 1)))

    for idx in nusc_utils.NUSC_LIDAR_STATIC_CLASSES:
        xx, yy = np.where(bev_seg_map[idx])
        xy = np.hstack((xx.reshape(-1, 1), yy.reshape(-1, 1)))
        c = bev_seg_map[idx, xx, yy].flatten()
        intp_cls = scipy.interpolate.griddata(
            points=xy,
            values=c,
            xi=xi,
            method='linear'
        )
        bev_seg_map[idx, :, :] = intp_cls.reshape(H, W)
        return bev_seg_map


class BEV:
    def __init__(self, nuscenes, sample):
        self.nuscenes = nuscenes
        self.sample = sample

        self.lidar_data = self.nuscenes.get('sample_data', self.sample['data']['LIDAR_TOP'])
        self.lidar_pcl = nusc_utils.load_point_cloud(self.nuscenes, self.lidar_data)
        self.lidar_seg = nusc_utils.load_lidarseg(nuscenes, self.sample['data']['LIDAR_TOP'])

        # merge multiple pedestrian class idx to one class
        self.cls_idx_merge_map, self.lidarseg_new_name2idx_mapping = nusc_utils.get_cls_mapping(nuscenes)
        self.idx_to_color = nusc_utils.get_idx_to_color(nuscenes)
        self.nusc_idx_to_color = {}
        for i, cls_label in enumerate(nusc_utils.NUSC_LIDAR_CLASS_NAMES):
            cls_idx = self.lidarseg_new_name2idx_mapping[cls_label]
            self.nusc_idx_to_color[i] = self.idx_to_color[cls_idx]

        self.intrinsics = None
        self.W = None
        self.H = None
        self.camera_data = None
        self.img_path = None
        self.img = None

        self.bev_projection_matrix = utils.projection_bev_matrix()

    def load_cam_data(self, cam):
        self.camera_data = self.nuscenes.get('sample_data', self.sample['data'][cam])
        self.H = self.camera_data['height']
        self.W = self.camera_data['width']
        self.intrinsics = nusc_utils.get_intrinsics(self.nuscenes, self.camera_data)
        self.img_path = self.nuscenes.dataroot / Path(self.camera_data['filename'])
        self.img = plt.imread(self.img_path)

    def project_lidar_to_image_plane(self):
        """

        :return: image_points_fov, cam_points_fov, idx
        image_points_fov - float64 numpy array - (n,2) - the coordinates of the lidar points on the image plane
        (in the image coordinate system uv: u - left to right, v - top to bottom)

        cam_points_fov - float64 numpy array - (n,2) - the coordinates of the lidar points in the camera
        coordinate system

        idx - the indices of the lidar points from the original lidar points array that are in the field of the view
        ex: lidar_pcl[idx] gives the coordinates of the lidar points in the lidar coordinate system that are in the
        field of view of the camera.
        """
        lidar_to_ego_frame_tfm = nusc_utils.sensor_to_ego_frame_tfm(self.nuscenes, self.lidar_data)
        camera_to_ego_frame_tfm = nusc_utils.sensor_to_ego_frame_tfm(self.nuscenes, self.camera_data)
        lidar_to_camera_tfm = np.linalg.inv(camera_to_ego_frame_tfm) @ lidar_to_ego_frame_tfm

        # project lidar point cloud to the camera coordinate system
        cam_points = utils.transform(lidar_to_camera_tfm, self.lidar_pcl)

        # only consider lidar points that are in front of the camera
        cam_front_mask = cam_points[:, 2] >= 0
        cam_points_front = cam_points[cam_front_mask]

        # project 3d points to the camera image plane
        image_points = utils.project_to_image_plane(self.intrinsics, cam_points_front)
        fov_mask = (image_points[:, 0] >= 0) & \
                   (image_points[:, 0] <= self.W) & \
                   (image_points[:, 1] >= 0) & \
                   (image_points[:, 1] <= self.H)
        cam_points_fov = cam_points_front[fov_mask]
        image_points_fov = image_points[fov_mask]
        idx = np.arange(self.lidar_pcl.shape[0])[cam_front_mask][fov_mask]

        return image_points_fov, cam_points_fov, idx

    def interpolate_depth(self, xy, z, method='linear'):
        """
        :param xy: image_points_fov
        :param z: cam_points_fov[:,2] # z coordinate (depth of lidar points)
        :param method: {'linear', 'nearest', 'cubic'}
        :return: interpolated depth
        """
        xx, yy = np.meshgrid(np.arange(self.H), np.arange(self.W))
        xi = np.hstack((yy.reshape(-1, 1), xx.reshape(-1, 1)))
        values = z
        intp_depth = scipy.interpolate.griddata(points=xy, values=values, xi=xi, method=method)
        depth = np.nan_to_num(intp_depth).reshape(self.W, self.H).T  # depth map
        return depth

    def interpolate_seg_cls(self, xy, c, method='linear'):
        """
        :param xy: image_points_fov
        :param c: the corresponding segmentation class index for each lidar point in the xy
        :param method: {'linear', 'nearest', 'cubic', 'knn'}
        :return:
        """

        xx, yy = np.meshgrid(np.arange(self.H), np.arange(self.W))
        xi = np.hstack((yy.reshape(-1, 1), xx.reshape(-1, 1)))

        # Class Interpolation
        # Use KNN to classify grid points based on given lidar segmentation
        if method == 'knn':
            neigh = KNeighborsClassifier(n_neighbors=5)
            neigh.fit(xy, c)
            intp_cls = neigh.predict(xi)
        else:
            intp_cls = scipy.interpolate.griddata(
                points=xy,
                values=c,
                xi=xi,
                method=method
            )

        # merge / remove classes to form a smaller set of classes
        cls_merged = np.array([self.cls_idx_merge_map.get(cls_idx, 0) for cls_idx in intp_cls])

        seg_cls = cls_merged.reshape(self.W, self.H).T

        return seg_cls

    def project_to_bev(self, u, z):
        """

        :param u: float64 numpy array - (n,) - the u coordinate of points in image coordinate system
        :param z: float64 numpy array - (n,) - the depth of the corresponding point in the camera coordinate system
        :return: 200x200 0/1 matrix representing the BEV grid
        where the grid cells are set to 1, when any point projected on to the BEV plane lies in the cell.

        BEV grid coordinate system is the image coordinate system UV where the origin of the camera coordinate system
        is at (100, 200)
        """
        # project to 3D space
        f, cu = self.intrinsics[0, 0], self.intrinsics[0, 2]
        x = z * (u - cu) / f
        bev_image_points = self.bev_projection_matrix @ np.array([x, z, np.ones_like(u)])
        bev_image_points = np.round(bev_image_points).astype(np.int_)

        # consider points which are in bev grid
        bev_mask = (bev_image_points[0, :] >= 0) & (bev_image_points[0, :] < 200) & \
                   (bev_image_points[1, :] >= 0) & (bev_image_points[1, :] < 200)

        bev_cls_map = np.zeros((200, 200))
        bev_cls_map[bev_image_points[1, :][bev_mask], bev_image_points[0, :][bev_mask]] = 1
        return bev_cls_map

    def create_seg_rgb(self, seg_map):
        seg_rgb = np.zeros((self.H, self.W, 3), dtype=np.uint8)
        for idx in set(np.unique(seg_map)) - {0}:
            ii, jj = np.where(seg_map == idx)
            seg_rgb[ii, jj, :] = np.array(self.idx_to_color[idx])
        return seg_rgb

    def visualize_depth_map(self, image_points_fov, cam_points_fov, depth_map, out_path=None):
        fig, axs = plt.subplots(nrows=2, figsize=(10, 10), dpi=300)
        ax = axs[0]

        ax.imshow(self.img)
        ax.scatter(image_points_fov[:, 0], image_points_fov[:, 1], c=cam_points_fov[:, 2], s=0.5,
                   cmap='plasma')
        ax.axis('off')

        ax = axs[1]
        ax.imshow(depth_map, cmap='plasma')
        ax.axis('off')

        fig.tight_layout()

        if out_path:
            fig.savefig(out_path, dpi=300)
        else:
            plt.show()

    def visualize_segmentation_map(self,
                                   image_points_fov,
                                   lidar_seg_fov,
                                   seg_map,
                                   out_path=None):
        fig, axs = plt.subplots(nrows=2, figsize=(10, 10), dpi=300)
        ax = axs[0]

        ax.imshow(self.img)
        ax.scatter(image_points_fov[:, 0], image_points_fov[:, 1], c=lidar_seg_fov, s=0.5,
                   cmap='plasma')
        ax.axis('off')

        seg_rgb = self.create_seg_rgb(seg_map=seg_map)

        img_class_idxs = sorted(list(set(np.unique(seg_map)) - {0}))
        img_class_labels = [
            nusc_utils.LABEL_RENAME[self.nuscenes.lidarseg_idx2name_mapping[idx]] for idx in img_class_idxs
        ]
        legend_colors = [np.append(np.array(self.idx_to_color[idx]) / 255, 1) for idx in img_class_idxs]
        patches = [mpatches.Patch(color=legend_colors[i], label=label)
                   for i, label in enumerate(img_class_labels)]

        ax = axs[1]
        ax.imshow(seg_rgb)
        ax.legend(handles=patches, bbox_to_anchor=(1.01, 1), loc=2, borderaxespad=0.)
        ax.axis('off')

        fig.tight_layout()

        if out_path:
            fig.savefig(out_path, dpi=300, bbox_inches='tight')
        else:
            plt.show()

    def interpolate_depth_seg(self, cam='CAM_FRONT', depth_intp_method='linear', seg_cls_intp_method='linear'):
        self.load_cam_data(cam)

        image_points_fov, cam_points_fov, idx = self.project_lidar_to_image_plane()
        depth_map = self.interpolate_depth(xy=image_points_fov, z=cam_points_fov[:, 2], method=depth_intp_method)

        lidar_seg_fov = self.lidar_seg[idx]
        _seg_cls = self.interpolate_seg_cls(xy=image_points_fov, c=lidar_seg_fov, method=seg_cls_intp_method)
        # only keep lidar points where depth is interpolated (not NaN)
        seg_map = np.where(np.isnan(depth_map), 0, _seg_cls)

        return depth_map, seg_map

    def generate_BEV_projection(self, depth_map, seg_map):
        bev_seg_map = np.zeros((14, 200, 200))
        for i, cls_label in enumerate(nusc_utils.NUSC_LIDAR_CLASS_NAMES):
            cls_idx = self.lidarseg_new_name2idx_mapping[cls_label]
            v, u = np.where(seg_map == cls_idx)
            z = depth_map[seg_map == cls_idx]
            bev_seg_map[i, :, :] = self.project_to_bev(u, z)

        return bev_seg_map


def generate_bev_seg_map(nuscenes, sample, depth_intp_method='linear', seg_cls_intp_method='linear', plot_results=False,
                         results_dir=None):
    bev = BEV(nuscenes, sample)
    bev.load_cam_data(cam='CAM_FRONT')

    image_points_fov, cam_points_fov, idx = bev.project_lidar_to_image_plane()
    depth_map = bev.interpolate_depth(xy=image_points_fov, z=cam_points_fov[:, 2], method=depth_intp_method)

    lidar_seg_fov = bev.lidar_seg[idx]
    _seg_cls = bev.interpolate_seg_cls(xy=image_points_fov, c=lidar_seg_fov, method=seg_cls_intp_method)
    # only keep lidar points where depth is interpolated (not NaN)
    seg_map = np.where(np.isnan(depth_map), 0, _seg_cls)

    # depth_map, seg_map = bev.interpolate_depth_seg(
    #     depth_intp_method=depth_intp_method,
    #     seg_cls_intp_method=seg_cls_intp_method
    # )
    if plot_results:
        bev.visualize_depth_map(image_points_fov, cam_points_fov, depth_map,
                                out_path=str(results_dir / 'depth.png'))
        bev.visualize_segmentation_map(image_points_fov, lidar_seg_fov, seg_map,
                                       out_path=str(results_dir / 'segmentation.png'))

    bev_seg_map = bev.generate_BEV_projection(depth_map, seg_map)
    # bev_seg_map_intp = interpolate_static_classes_bev(bev_seg_map)
    return bev_seg_map, bev.nusc_idx_to_color


def main():
    dataroot = Path('/Users/deepakduggirala/Documents/autonomous-robotics/v1.0-mini/').resolve()
    nuscenes = NuScenes(version='v1.0-mini', dataroot=str(dataroot), verbose=False)
    scene_idx = 0
    sample_idx = 30

    scene = nuscenes.scene[scene_idx]
    samples = list(nusc_utils.sample_gen(nuscenes, scene))
    sample = samples[sample_idx]

    results_dir = Path(f'results/scene-{scene_idx}/sample-{sample_idx}').resolve()
    results_dir.mkdir(parents=True, exist_ok=True)

    bev = BEV(nuscenes, sample)
    bev.load_cam_data(cam='CAM_FRONT')

    image_points_fov, cam_points_fov, idx = bev.project_lidar_to_image_plane()
    depth_map = bev.interpolate_depth(xy=image_points_fov, z=cam_points_fov[:, 2])
    bev.visualize_depth_map(image_points_fov, cam_points_fov, depth_map)

    lidar_seg_fov = bev.lidar_seg[idx]
    _seg_cls = bev.interpolate_seg_cls(xy=image_points_fov, c=lidar_seg_fov)
    # only keep lidar points where depth is interpolated (not NaN)
    seg_map = np.where(np.isnan(depth_map), 0, _seg_cls)
    bev.visualize_segmentation_map(image_points_fov, lidar_seg_fov, seg_map)
