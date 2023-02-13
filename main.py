import sys

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from pathlib import Path

import cv2
import numpy as np
from nuscenes.nuscenes import NuScenes

from bev_gt import generate_gt_bev_map
from bev2 import generate_bev_seg_map

import utils
import nusc_utils


def main(scene_idx, sample_idx):
    # Load NuScenes dataset
    dataroot = '/Users/deepakduggirala/Documents/autonomous-robotics/v1.0-mini'  # os.path.expandvars(config.dataroot)
    nuscenes = NuScenes('v1.0-mini', dataroot, verbose=False)

    scene = nuscenes.scene[scene_idx]
    samples = list(nusc_utils.sample_gen(nuscenes, scene))
    sample = samples[sample_idx]

    results_dir = Path(f'results-linear-linear-static/scene-{scene_idx}/sample-{sample_idx}').resolve()
    results_dir.mkdir(parents=True, exist_ok=True)

    bev_gt_map = generate_gt_bev_map(nuscenes, scene, sample)
    bev_seg_map, nusc_idx_to_color = generate_bev_seg_map(nuscenes, sample,
                                                          seg_cls_intp_method='linear',
                                                          plot_results=True,
                                                          results_dir=results_dir)

    bev_gt_map_cmp = utils.make_composite(bev_gt_map)
    bev_seg_map_cmp = utils.make_composite(bev_seg_map)

    color_map = {i + 1: c for i, c in nusc_idx_to_color.items()}
    color_map[0] = [255, 255, 255]
    color_map[2] = [255, 255, 255]  # do not plot "other_flat"
    color_map[4] = [255, 255, 255]  # do not plot "terrain"
    color_map[15] = [255, 255, 255]  # do not plot "lidar mask"

    bev_gt_map_cmp_color = utils.color_components(bev_gt_map_cmp, color_map=color_map)
    bev_seg_map_cmp_color = utils.color_components(bev_seg_map_cmp, color_map=color_map)

    fig, axs = plt.subplots(ncols=2, figsize=(15, 7))
    ax = axs[0]
    ax.imshow(cv2.flip(bev_gt_map_cmp_color, 0))
    ax.set_title('map based BEV')
    ax.axis('off')

    ax = axs[1]
    ax.imshow(bev_seg_map_cmp_color)
    ax.set_title('lidarseg BEV')

    legend_colors = [np.append(np.array(nusc_idx_to_color[idx]) / 255, 1) for idx in range(len(nusc_idx_to_color))]
    patches = [mpatches.Patch(color=legend_colors[i], label=label)
               for i, label in enumerate(nusc_utils.NUSC_LIDAR_CLASS_NAMES) if i not in [1, 3]]
    ax.legend(handles=patches, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    ax.axis('off')

    out_path = str(results_dir / 'bev.png')
    fig.savefig(out_path, dpi=300, bbox_inches='tight')


if __name__ == '__main__':
    scene_idx = int(sys.argv[1])
    sample_idx = int(sys.argv[2])
    print(f'generating bevs for scene {scene_idx}, sample {sample_idx}')
    main(scene_idx, sample_idx)
