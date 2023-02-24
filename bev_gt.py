from collections import OrderedDict, namedtuple

import numpy as np

from shapely.strtree import STRtree
from nuscenes.nuscenes import NuScenes
from nuscenes.map_expansion.map_api import NuScenesMap

import utils
import nusc_utils


def process_sample_data(nuscenes, map_data, sample_data, lidar, config):
    # Render static road geometry masks
    map_masks = nusc_utils.get_map_masks(nuscenes,
                                         map_data,
                                         sample_data,
                                         config.map_extents,
                                         config.map_resolution)

    # Render dynamic object masks
    obj_masks = nusc_utils.get_object_masks(nuscenes,
                                            sample_data,
                                            config.map_extents,
                                            config.map_resolution)
    masks = np.concatenate([map_masks, obj_masks], axis=0)

    # Ignore regions of the BEV which are outside the image
    sensor = nuscenes.get('calibrated_sensor',
                          sample_data['calibrated_sensor_token'])
    intrinsics = np.array(sensor['camera_intrinsic'])
    masks[-1] |= ~utils.get_visible_mask(intrinsics, sample_data['width'],
                                         config.map_extents, config.map_resolution)

    # Transform lidar points into camera coordinates
    cam_transform = nusc_utils.get_sensor_transform(nuscenes, sample_data)
    cam_points = utils.transform(np.linalg.inv(cam_transform), lidar)
    masks[-1] |= utils.get_occlusion_mask(cam_points, config.map_extents,
                                          config.map_resolution)

    return masks
    # # Encode masks as integer bitmask
    # labels = encode_binary_labels(masks)

    # # Save outputs to disk
    # output_path = os.path.join(os.path.expandvars(config.label_root),
    #                            sample_data['token'] + '.png')
    # Image.fromarray(labels.astype(np.int32), mode='I').save(output_path)


def process_sample_data_without_vis_mask(nuscenes, map_data, sample_data, config):
    # Render static road geometry masks
    map_masks = nusc_utils.get_map_masks(nuscenes,
                                         map_data,
                                         sample_data,
                                         config.map_extents,
                                         config.map_resolution)

    # Render dynamic object masks
    obj_masks = nusc_utils.get_object_masks(nuscenes,
                                            sample_data,
                                            config.map_extents,
                                            config.map_resolution)
    masks = np.concatenate([map_masks, obj_masks], axis=0)

    return masks


def load_map_data(dataroot, location):
    # Load the NuScenes map object
    nusc_map = NuScenesMap(dataroot, location)

    map_data = OrderedDict()
    for layer in nusc_utils.STATIC_CLASSES:

        # Retrieve all data associated with the current layer
        records = getattr(nusc_map, layer)
        polygons = list()

        # Drivable area records can contain multiple polygons
        if layer == 'drivable_area':
            for record in records:

                # Convert each entry in the record into a shapely object
                for token in record['polygon_tokens']:
                    poly = nusc_map.extract_polygon(token)
                    if poly.is_valid:
                        polygons.append(poly)
        else:
            for record in records:

                # Convert each entry in the record into a shapely object
                poly = nusc_map.extract_polygon(record['polygon_token'])
                if poly.is_valid:
                    polygons.append(poly)

        # Store as an R-Tree for fast intersection queries
        map_data[layer] = STRtree(polygons)

    return map_data


def generate_gt_bev_map(nuscenes, scene, sample):
    Config = namedtuple('Config', ['map_extents', 'map_resolution'])
    config = Config(map_extents=[-25., 1., 25., 50.], map_resolution=0.25)

    # Preload NuScenes map data
    map_data = {
        location: load_map_data(nuscenes.dataroot, location)
        for location in nusc_utils.LOCATIONS
    }

    # scene = nuscenes.scene[scene_idx]
    # samples = list(nusc_utils.sample_gen(nuscenes, scene))
    # sample = samples[sample_idx]

    log = nuscenes.get('log', scene['log_token'])
    scene_map_data = map_data[log['location']]
    # process_sample(nuscenes, scene_map_data, sample, config)

    # Load the lidar point cloud associated with this sample
    lidar_data = nuscenes.get('sample_data', sample['data']['LIDAR_TOP'])
    lidar_pcl = nusc_utils.load_point_cloud(nuscenes, lidar_data)

    # Transform points into world coordinate system
    lidar_transform = nusc_utils.get_sensor_transform(nuscenes, lidar_data)
    lidar_pcl = utils.transform(lidar_transform, lidar_pcl)

    camera_data = nuscenes.get('sample_data', sample['data']['CAM_FRONT'])
    return process_sample_data(nuscenes, scene_map_data, camera_data, lidar_pcl, config)


def generate_gt_bev_360_maps_scene(nuscenes, scene):
    log = nuscenes.get('log', scene['log_token'])
    location = log['location']
    scene_map_data = load_map_data(nuscenes.dataroot, location)
    bev_maps = []
    for sample in nusc_utils.sample_gen(nuscenes, scene):
        bev_map = generate_gt_bev_360_map_sample(nuscenes, sample, scene_map_data)
        bev_maps.append(bev_map)
    return bev_maps


def generate_gt_bev_360_map_sample(nuscenes, sample, scene_map_data):
    Config = namedtuple('Config', ['map_extents', 'map_resolution'])
    config = Config(map_extents=[-25., -50., 25., 50.], map_resolution=0.25)
    camera_data = nuscenes.get('sample_data', sample['data']['CAM_FRONT'])
    return process_sample_data_without_vis_mask(nuscenes, scene_map_data, camera_data, config)
