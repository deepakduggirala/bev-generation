import time
import zlib
from pathlib import Path

import lmdb
from nuscenes.nuscenes import NuScenes

import nusc_utils
from bev_gt import generate_gt_bev_360_map_sample, load_map_data
from utils import array_to_bytes

import warnings

warnings.filterwarnings("ignore")


def main(data_root, version):
    start = time.perf_counter()

    nuscenes = NuScenes(version=version, dataroot=str(data_root), verbose=False)
    print('v1.0-trainval loaded', int(time.perf_counter() - start), 'sec')

    db_path = data_root / Path(f'lmdb/samples/GT_BEV_360')
    db_path.mkdir(parents=True, exist_ok=True)
    print('writing to', db_path)

    db_map_size = int(15 * 1024 * 1024 * 1024)

    with lmdb.open(path=str(db_path), map_size=db_map_size) as lmdb_env:
        for i, scene in enumerate(nuscenes.scene, 1):
            print('processing scene', i)
            log = nuscenes.get('log', scene['log_token'])
            location = log['location']
            scene_map_data = load_map_data(nuscenes.dataroot, location)
            with lmdb_env.begin(write=True) as write_txn:
                for sample in nusc_utils.sample_gen(nuscenes, scene):
                    bev_gt_map = generate_gt_bev_360_map_sample(nuscenes, sample, scene_map_data)
                    key = bytes(sample['token'], 'utf-8')
                    value = array_to_bytes(bev_gt_map)
                    value_zipped = zlib.compress(value)
                    write_txn.put(key, value_zipped)


# main(Path.resolve(Path('/N/slate/deduggi/nuScenes-trainval')), 'v1.0-trainval')
main(Path.resolve(Path('/Users/deepakduggirala/Documents/autonomous-robotics/v1.0-mini/')), 'v1.0-mini')
