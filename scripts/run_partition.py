
import sys, os, json, time
sys.path.append(os.path.expanduser('~/superpoint_transformer'))

import torch
import numpy as np
import laspy
from copy import deepcopy
from src.data import Data
from src.utils import init_config
from src.utils.color import to_float_rgb
from src.transforms import *

# Reader function
def read_vancouver_tile(
        filepath, 
        xyz=True, 
        rgb=True, 
        intensity=True, 
        semantic=True, 
        instance=False,
        remap=True, 
        max_intensity=600):
    """Read a Vancouver tile saved as LAS.

    :param filepath: str
        Absolute path to the LAS file
    :param xyz: bool
        Whether XYZ coordinates should be saved in the output Data.pos
    :param rgb: bool
        Whether RGB colors should be saved in the output Data.rgb
    :param intensity: bool
        Whether intensity should be saved in the output Data.rgb
    :param semantic: bool
        Whether semantic labels should be saved in the output Data.y
    :param instance: bool
        Whether instance labels should be saved in the output Data.obj
    :param remap: bool
        Whether semantic labels should be mapped from their Vancouver ID
        to their train ID
    :param max_intensity: float
        Maximum value used to clip intensity signal before normalizing 
        to [0, 1]
    """
    # Create an emty Data object
    data = Data()
    
    las = laspy.read(filepath)

    # Populate data with point coordinates 
    if xyz:
        # Apply the scale provided by the LAS header
        pos = torch.stack([
            torch.tensor(las[axis])
            for axis in ["X", "Y", "Z"]], dim=-1)
        pos *= las.header.scale
        pos_offset = pos[0]
        data.pos = (pos - pos_offset).float()
        data.pos_offset = pos_offset

    # Populate data with point RGB colors
    if rgb:
        # RGB stored in uint16 lives in [0, 65535]
        data.rgb = to_float_rgb(torch.stack([
            torch.FloatTensor(las[axis].astype('float32') / 65535)
            for axis in ["red", "green", "blue"]], dim=-1))

    # Populate data with point LiDAR intensity
    if intensity:
        # Heuristic to bring the intensity distribution in [0, 1]
        data.intensity = torch.FloatTensor(
            las['intensity'].astype('float32')
        ).clip(min=0, max=max_intensity) / max_intensity

    # Populate data with point semantic segmentation labels
    if semantic:
        y = torch.LongTensor(las['classification'])
        data.y = torch.from_numpy(ID2TRAINID)[y] if remap else y

    # Populate data with point panoptic segmentation labels
    if instance:
        raise NotImplementedError("The dataset does not contain instance labels.")

    return data


# Config
VANCOUVER_NUM_CLASSES = 6
ID2TRAINID = np.asarray([VANCOUVER_NUM_CLASSES, 5, 0, 3, VANCOUVER_NUM_CLASSES, 2, 4, VANCOUVER_NUM_CLASSES, VANCOUVER_NUM_CLASSES, 1])
DALES_NUM_CLASSES = 8
VANCOUVER_TO_DALES = np.asarray([0, DALES_NUM_CLASSES, 1, 1, 7, DALES_NUM_CLASSES, DALES_NUM_CLASSES])

# Read args
reg = [float(x) for x in sys.argv[1:4]]
filepath = sys.argv[4]
ckpt_path = sys.argv[5]

# Init transforms
cfg = init_config(overrides=["experiment=semantic/dales"])
transforms_dict = instantiate_datamodule_transforms(cfg.datamodule)

# Read and tile data
data_raw = read_vancouver_tile(filepath)
data_raw = SampleXYTiling(x=1, y=1, tiling=3)(data_raw)

# Override regularization in pre_transform
pre_t = deepcopy(transforms_dict['pre_transform'])
for t in pre_t.transforms:
    if isinstance(t, CutPursuitPartition):
        t.regularization = reg

# Partition
t0 = time.time()
nag_i = pre_t(data_raw)
partition_time = time.time() - t0

num_sp_l1 = nag_i[1].num_points
oracle = nag_i[1].semantic_segmentation_oracle(VANCOUVER_NUM_CLASSES)
oracle_miou = oracle['miou'].item()

# Load model
import hydra
model = hydra.utils.instantiate(cfg.model)
model = model._load_from_checkpoint(ckpt_path)

# Prepare for inference
nag_inf = nag_i.clone()
del nag_i, data_raw
nag_inf = NAGRemoveKeys(level=0, keys=[k for k in nag_inf[0].keys if k not in cfg.datamodule.point_load_keys])(nag_inf)
nag_inf = NAGRemoveKeys(level='1+', keys=[k for k in nag_inf[1].keys if k not in cfg.datamodule.segment_load_keys])(nag_inf)
nag_inf = nag_inf.cuda()
nag_inf = transforms_dict['on_device_test_transform'](nag_inf)

# Inference
model = model.eval().cuda()
with torch.no_grad():
    output = model(nag_inf)

preds = output.voxel_semantic_pred(super_index=nag_inf[0].super_index).cpu().numpy()
gt_hist = nag_inf[0].y.cpu()
vancouver_labels = gt_hist.argmax(dim=1).numpy()
ground_truth = VANCOUVER_TO_DALES[vancouver_labels]

mask = ground_truth < DALES_NUM_CLASSES
gt_valid = ground_truth[mask]
pred_valid = preds[mask]

ious = {}
for cls_id, cls_name in [(0, "Ground"), (1, "Vegetation"), (7, "Buildings")]:
    inter = ((pred_valid == cls_id) & (gt_valid == cls_id)).sum()
    union = ((pred_valid == cls_id) | (gt_valid == cls_id)).sum()
    ious[cls_name] = float(inter / union) if union > 0 else 0.0

actual_miou = float(np.mean(list(ious.values())))

# Output as JSON on last line
print(json.dumps({
    "regularization": reg[0],
    "num_sp_l1": int(num_sp_l1),
    "oracle_miou": float(oracle_miou),
    "actual_miou": actual_miou,
    "partition_time": float(partition_time),
    "ious": ious
}))
