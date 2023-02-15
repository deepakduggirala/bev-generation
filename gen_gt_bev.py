import time
import zlib
from pathlib import Path

import lmdb
from nuscenes.nuscenes import NuScenes

import nusc_utils
from bev_gt import generate_gt_bev_map
from utils import array_to_bytes


def main():
    start = time.perf_counter()

    data_root = Path.resolve(Path('/N/slate/deduggi/nuScenes-trainval'))
    nuscenes = NuScenes(version='v1.0-trainval', dataroot=str(data_root), verbose=False)
    print('v1.0-trainval loaded', int(time.perf_counter() - start), 'sec')

    db_path = data_root / Path(f'lmdb/samples/GT_CAM_FRONT')
    db_path.mkdir(parents=True, exist_ok=True)
    print('writing to', db_path)

    db_map_size = int(15 * 1024 * 1024 * 1024)

    with lmdb.open(path=db_path, map_size=db_map_size) as lmdb_env:
        for i, scene in enumerate(nuscenes.scene, 1):
            print('starting scene', i)
            start = time.perf_counter()
            process_scene(lmdb_env, nuscenes, scene)
            print('completed scene', i, 'time took', time.perf_counter() - start)
            print('\n\n\n')


def process_scene(lmdb_env, nuscenes, scene):
    with lmdb_env.begin(write=True) as write_txn:
        for i, sample in enumerate(nusc_utils.sample_gen(nuscenes, scene),1):
            bev_gt_map = generate_gt_bev_map(nuscenes, scene, sample)
            key = bytes(sample['data']['CAM_FRONT'], 'utf-8')
            value = array_to_bytes(bev_gt_map)
            value_zipped = zlib.compress(value)
            write_txn.put(key, value_zipped)
            print('\t\t\tcompleted sample', i)