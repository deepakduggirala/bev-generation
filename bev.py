import os
from pathlib import Path
import itertools

import numpy as np
import scipy

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from sklearn.neighbors import KNeighborsClassifier

from nuscenes.nuscenes import NuScenes

import utils
import nusc_utils


def project_lidar_to_image_plane(nuscenes, sample, cam='CAM_FRONT'):
    lidar_data = nuscenes.get('sample_data', sample['data']['LIDAR_TOP'])
    lidar_pcl = nusc_utils.load_point_cloud(nuscenes, lidar_data)

    camera_data = nuscenes.get('sample_data', sample['data'][cam])
    H = camera_data['height']
    W = camera_data['width']
    intrinsics = nusc_utils.get_intrinsics(nuscenes, camera_data)

    lidar_to_ego_frame_tfm = nusc_utils.sensor_to_ego_frame_tfm(nuscenes, lidar_data)
    camera_to_ego_frame_tfm = nusc_utils.sensor_to_ego_frame_tfm(nuscenes, camera_data)
    lidar_to_camera_tfm = np.linalg.inv(camera_to_ego_frame_tfm) @ lidar_to_ego_frame_tfm

    # project lidar point cloud to the camera coordinate system
    cam_points = utils.transform(lidar_to_camera_tfm, lidar_pcl)
    # only consider lidar points that are in front of the camera
    cam_front_mask = cam_points[:, 2] >= 0
    cam_points_front = cam_points[cam_front_mask]

    # project 3d points to the camera image plane
    image_points = utils.project_to_image_plane(intrinsics, cam_points_front)
    fov_mask = (image_points[:, 0] >= 0) & \
               (image_points[:, 0] <= W) & \
               (image_points[:, 1] >= 0) & \
               (image_points[:, 1] <= H)
    cam_points_fov = cam_points_front[fov_mask]
    image_points_fov = image_points[fov_mask]
    idx = np.arange(lidar_pcl.shape[0])[cam_front_mask][fov_mask]

    return image_points_fov, cam_points_fov, idx


def interpolate_depth(img_shape, xy, z):
    """
    :param img_shape: (H, W)
    :param xy: image_points_fov
    :param z: cam_points_fov[:,2] # z coordinate (depth of lidar points)
    :return: interpolated depth
    """
    H, W = img_shape
    xx, yy = np.meshgrid(np.arange(H), np.arange(W))
    xi = np.hstack((yy.reshape(-1, 1), xx.reshape(-1, 1)))
    values = z
    intp_depth = scipy.interpolate.griddata(points=xy, values=values, xi=xi, method='linear')
    depth = np.nan_to_num(intp_depth).reshape(W, H).T  # depth map
    return depth


def interpolate_seg_cls(img_shape, xy, c, old_cls_idx2new_cls_idx):
    """
    :param img_shape: (H, W)
    :param xy: image_points_fov
    :param c: the corresponding segmentation class index for each lidar point in the xy
    :param old_cls_idx2new_cls_idx:
    :return:
    """
    H, W = img_shape
    xx, yy = np.meshgrid(np.arange(H), np.arange(W))
    xi = np.hstack((yy.reshape(-1, 1), xx.reshape(-1, 1)))
    intp_cls = scipy.interpolate.griddata(
        points=xy,
        values=c,
        xi=xi,
        method='linear'
    )

    # merge / remove classes to form a smaller set of classes
    cls_merged = np.array([old_cls_idx2new_cls_idx.get(cls_idx, 0) for cls_idx in intp_cls])

    seg_cls = cls_merged.reshape(W, H).T

    return seg_cls


def create_seg_rgb(img_shape, seg_cls, idx_to_color):
    H, W = img_shape
    seg_rgb = np.zeros((H, W, 3), dtype=np.uint8)
    for idx in set(np.unique(seg_cls)) - {0}:
        ii, jj = np.where(seg_cls == idx)
        seg_rgb[ii, jj, :] = np.array(idx_to_color[idx])
    return seg_rgb


# noinspection PyTypeChecker
def generate_bev_seg_map(nuscenes, scene_idx, sample_idx):
    scene = nuscenes.scene[scene_idx]
    samples = list(nusc_utils.sample_gen(nuscenes, scene))
    sample = samples[sample_idx]
    lidar_data = nuscenes.get('sample_data', sample['data']['LIDAR_TOP'])
    camera_data = nuscenes.get('sample_data', sample['data']['CAM_FRONT'])
    lidar_seg = nusc_utils.load_lidarseg(nuscenes, sample['data']['LIDAR_TOP'])
    img_path = nuscenes.dataroot / Path(camera_data['filename'])
    img = plt.imread(img_path)

    results_dir = Path(f'results-2/scene-{scene_idx}/sample-{sample_idx}').resolve()
    results_dir.mkdir(parents=True, exist_ok=True)

    lidar_pcl = nusc_utils.load_point_cloud(nuscenes, lidar_data)
    lidar_to_ego_frame_tfm = nusc_utils.sensor_to_ego_frame_tfm(nuscenes, lidar_data)
    camera_to_ego_frame_tfm = nusc_utils.sensor_to_ego_frame_tfm(nuscenes, camera_data)
    lidar_to_camera_tfm = np.linalg.inv(camera_to_ego_frame_tfm) @ lidar_to_ego_frame_tfm

    # project lidar point cloud to the camera coordinate system
    cam_points = utils.transform(lidar_to_camera_tfm, lidar_pcl)

    # only consider lidar points that are in front of the camera
    cam_front_mask = cam_points[:, 2] >= 0
    cam_points_front = cam_points[cam_front_mask]
    intrinsics = nusc_utils.get_intrinsics(nuscenes, camera_data)

    # project 3d points to the camera image plane
    image_points = utils.project_to_image_plane(intrinsics, cam_points_front)

    # create a field of view mask (fov) (True for points which are in the camera fov)
    H = camera_data['height']
    W = camera_data['width']
    fov_mask = (image_points[:, 0] >= 0) & \
               (image_points[:, 0] <= W) & \
               (image_points[:, 1] >= 0) & \
               (image_points[:, 1] <= H)
    cam_points_fov = cam_points_front[fov_mask]
    image_points_fov = image_points[fov_mask]
    idx = np.arange(lidar_pcl.shape[0])[cam_front_mask][fov_mask]

    # Depth Interpolation
    # generate grid points to interpolate depth
    print('interpolating depth')
    xi = np.array(list(itertools.product(np.arange(0, W), np.arange(0, H))))
    values = cam_points_fov[:, 2]  # z coordinate (depth of lidar points)
    intp_depth = scipy.interpolate.griddata(points=image_points_fov, values=values, xi=xi, method='linear')
    depth = np.nan_to_num(intp_depth).reshape(W, H).T  # depth map
    visualize_depth_map(img, cam_points_fov, image_points_fov, depth, out_path=str(results_dir / 'depth.png'))

    # Class Interpolation
    # Use KNN to classify grid points based on given lidar segmentation
    print('interpolating segmentation classes')
    lidar_seg_fov = lidar_seg[idx]
    # neigh = KNeighborsClassifier(n_neighbors=5)
    # neigh.fit(image_points_fov, lidar_seg_fov)
    # intp_cls = neigh.predict(xi)

    intp_cls = scipy.interpolate.griddata(
        points=image_points_fov,
        values=lidar_seg_fov,
        xi=xi,
        method='linear'
    )

    # # only keep lidar points where depth is interpolated (not NaN)
    cls_avail_lidar = np.where(np.isnan(intp_depth), 0, intp_cls)

    # merge pedestrian classes and ignore rare classes
    lidarseg_new_name2idx_mapping = {}
    for old_cls, new_cls in nusc_utils.LABEL_RENAME.items():
        old_cls_idx = nuscenes.lidarseg_name2idx_mapping[old_cls]
        lidarseg_new_name2idx_mapping[new_cls] = old_cls_idx
    old_cls_idx2new_cls_idx = {nuscenes.lidarseg_name2idx_mapping[old_cls]: lidarseg_new_name2idx_mapping[new_cls] for
                               old_cls, new_cls in nusc_utils.LABEL_RENAME.items()}
    idx_to_color = {nuscenes.lidarseg_name2idx_mapping[cls_name]: rgb for cls_name, rgb in nuscenes.colormap.items()}

    # merge / remove classes to form a smaller set of classes
    cls_selected = np.array([old_cls_idx2new_cls_idx.get(cls_idx, 0) for cls_idx in cls_avail_lidar])

    seg_cls = cls_selected.reshape(W, H).T
    seg_rgb = np.zeros((900, 1600, 3), dtype=np.uint8)
    for idx in set(np.unique(seg_cls)) - {0}:
        ii, jj = np.where(seg_cls == idx)
        seg_rgb[ii, jj, :] = np.array(idx_to_color[idx])
    visualize_segmentation_map(nuscenes, img, lidar_seg_fov, image_points_fov, seg_cls, seg_rgb,
                               idx_to_color, out_path=str(results_dir / 'segmentation.png'))

    # BEV projection
    print('generating BEV projection')

    bev_projection_matrix = utils.projection_bev_matrix()
    f, cu, cv = intrinsics[0, 0], intrinsics[0, 2], intrinsics[1, 2]

    def project_to_bev(u, z):
        # project to 3D space
        x = z * (u - cu) / f
        bev_image_points = bev_projection_matrix @ np.array([x, z, np.ones_like(u)])
        bev_image_points = np.round(bev_image_points).astype(np.int_)

        # consider points which are in bev grid
        bev_mask = (bev_image_points[0, :] >= 0) & (bev_image_points[0, :] < 200) & \
                   (bev_image_points[1, :] >= 0) & (bev_image_points[1, :] < 200)

        bev_cls = np.zeros((200, 200))
        bev_cls[bev_image_points[1, :][bev_mask], bev_image_points[0, :][bev_mask]] = 1
        return bev_cls

    bev_seg_map = np.zeros((14, 200, 200))
    nusc_idx_to_color = {}
    for i, cls_label in enumerate(nusc_utils.NUSC_LIDAR_CLASS_NAMES):
        cls_idx = lidarseg_new_name2idx_mapping[cls_label]
        nusc_idx_to_color[i] = idx_to_color[cls_idx]
        v, u = np.where(seg_cls == cls_idx)
        z = depth[seg_cls == cls_idx]
        bev_seg_map[i, :, :] = project_to_bev(u, z)

    return bev_seg_map, idx_to_color, nusc_idx_to_color


def visualize_depth_map(img, cam_points_fov, image_points_fov, depth, out_path=None):
    fig, axs = plt.subplots(nrows=2, figsize=(10, 10), dpi=300)
    ax = axs[0]

    ax.imshow(img)
    ax.scatter(image_points_fov[:, 0], image_points_fov[:, 1], c=cam_points_fov[:, 2], s=0.5,
               cmap='plasma')
    ax.axis('off')

    ax = axs[1]
    ax.imshow(depth, cmap='plasma')
    ax.axis('off')

    fig.tight_layout()

    if out_path:
        fig.savefig(out_path, dpi=300)
    else:
        plt.show()


def visualize_segmentation_map_old(nuscenes, img, lidar_seg_fov,
                               image_points_fov, seg_cls, seg_rgb,
                               idx_to_color, out_path=None):
    fig, axs = plt.subplots(nrows=2, figsize=(10, 10), dpi=300)
    ax = axs[0]

    ax.imshow(img)
    ax.scatter(image_points_fov[:, 0], image_points_fov[:, 1], c=lidar_seg_fov, s=0.5,
               cmap='plasma')
    ax.axis('off')

    img_class_idxs = sorted(list(set(np.unique(seg_cls)) - {0}))
    img_class_labels = [nusc_utils.LABEL_RENAME[nuscenes.lidarseg_idx2name_mapping[idx]] for idx in img_class_idxs]
    legend_colors = [np.append(np.array(idx_to_color[idx]) / 255, 1) for idx in img_class_idxs]
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


def visualize_segmentation_map(nuscenes, img,
                               image_points_fov, lidar_seg_fov,
                               seg_cls, idx_to_color, out_path=None):
    fig, axs = plt.subplots(nrows=2, figsize=(10, 10), dpi=300)
    ax = axs[0]

    ax.imshow(img)
    ax.scatter(image_points_fov[:, 0], image_points_fov[:, 1], c=lidar_seg_fov, s=0.5,
               cmap='plasma')
    ax.axis('off')

    seg_rgb = create_seg_rgb(img_shape=img.shape[:2], seg_cls=seg_cls, idx_to_color=idx_to_color)

    img_class_idxs = sorted(list(set(np.unique(seg_cls)) - {0}))
    img_class_labels = [nusc_utils.LABEL_RENAME[nuscenes.lidarseg_idx2name_mapping[idx]] for idx in img_class_idxs]
    legend_colors = [np.append(np.array(idx_to_color[idx]) / 255, 1) for idx in img_class_idxs]
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


def main():
    dataroot = Path('/Users/deepakduggirala/Documents/autonomous-robotics/v1.0-mini/').resolve()
    nuscenes = NuScenes(version='v1.0-mini', dataroot=str(dataroot), verbose=False)
    scene_idx = 0
    sample_idx = 30

    generate_bev_seg_map(nuscenes, scene_idx, sample_idx)


if __name__ == '__main__':
    main()
