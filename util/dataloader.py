import torch
from glob import glob
import os
import numpy as np
import torch.nn.functional as F
import pickle
import h5py
from PIL import Image
import math
from sklearn.model_selection import train_test_split
from torchvision import transforms
from cachetools import LRUCache
import h5py
import mat73
import json
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
from torchvision.io import read_image, ImageReadMode
from scipy.io import loadmat

def safe_read_image(path):
    try:
        img = read_image(path, mode=ImageReadMode.RGB)
    except Exception as e:
        print(f"[Warning] Failed to load {path} with torchvision ({e}). Falling back to PIL.")
        img = Image.open(path).convert('RGB')
        img = torch.from_numpy(np.array(img)).permute(2, 0, 1)
    return img

class MemmapLRU(LRUCache):
    def __init__(self, maxsize, **kwargs):
        super().__init__(maxsize, **kwargs)

    def popitem(self):
        # this is called by LRUCache whenever it needs to evict
        key, mm = super().popitem()
        mm._mmap.close()    # close the file-descriptor
        return key, mm
    
# allow up to maxsize open memmaps per worker
_memmap_feat_cached = MemmapLRU(maxsize=130)

def _load_mmap(vid_name, feat_path=None):
    if vid_name not in _memmap_feat_cached:
        mm = np.load(os.path.join(feat_path, vid_name), mmap_mode='r')
        _memmap_feat_cached[vid_name] = mm

    return _memmap_feat_cached[vid_name]

def worker_init_fn(worker_id):
    global _memmap_feat_cached
    _memmap_feat_cached = MemmapLRU(maxsize=130)


class COCO_feat(torch.utils.data.Dataset):
    """
    Dataset for training prototypes on COCO data.
    """
    def __init__(self, feature_dir, split='train'):
        self.feature_dir = feature_dir
        self.split = split

        self.all_img = glob(os.path.join(self.feature_dir, '*.npy'))
        self.all_img = [cur for cur in self.all_img if split in os.path.basename(cur)]

        print("%d Image in total used for %s" %(len(self.all_img), self.split))

    def __getitem__(self, index):
        cur_feat_path = self.all_img[index]
        img_name = os.path.basename(cur_feat_path)[:-4]

        # load feature
        img_feat = _load_mmap(img_name+'.npy', os.path.dirname(cur_feat_path))

        return img_feat, img_name 

    def __len__(self):
        return len(self.all_img)
    

class NSD_feat(torch.utils.data.Dataset):
    """
    Dataset for loading Dino features for NSD images
    """
    def __init__(self, feature_dir, ):
        self.feature_dir = feature_dir
        all_subj = glob(os.path.join(self.feature_dir, 'subj*'))
        self.all_img = []
        self.all_subj = []
        for subj_dir in all_subj:
            img_paths = glob(os.path.join(subj_dir, '*.npy'))
            self.all_img.extend(img_paths)
            self.all_subj.extend([os.path.basename(subj_dir)]*len(img_paths))
        print("%d Image in total found" %(len(self.all_img)))

    def __getitem__(self, index):
        cur_feat_path = self.all_img[index]
        img_name = os.path.basename(cur_feat_path)[:-4]
        subj_name = self.all_subj[index]
        # load feature
        img_feat = _load_mmap(img_name+'.npy', os.path.dirname(cur_feat_path))

        return img_feat, img_name, subj_name
    def __len__(self):
        return len(self.all_img)

class NSD_proto(torch.utils.data.Dataset):
    """
    Dataset for learning the mapping between prototype similarity and voxel activations for NSD data. 
    """
    def __init__(self, proto_dir, nsd_dir, subj_name, area_name, patch_name, seed=42, split='train'):
        
        raw_data = []
        for hemi in ['lh', 'rh']:
            cur_data = np.load(os.path.join(nsd_dir, subj_name, 
                            'training_split', 'training_fmri', 
                             f'{hemi}_training_fmri.npy'))
            cur_roi = np.load(os.path.join(nsd_dir, subj_name, 
                            'roi_masks', 
                             hemi+'.floc-'+area_name+'_challenge_space.npy'))
            cur_mapping = np.load(os.path.join(nsd_dir, subj_name, 
                            'roi_masks', 
                             'mapping_floc-'+area_name+'.npy'), allow_pickle=True).item()
            if patch_name == None: 
                valid_voxels = np.where(cur_roi>0)[0]
            else:
                patch2id = {cur_mapping[k]:k for k in cur_mapping.keys()}
                assert patch_name in patch2id, f"Patch name {patch_name} not found in mapping for area {area_name}"
                patch_id = patch2id[patch_name]
                valid_voxels = np.where(cur_roi==patch_id)[0]

            raw_data.append(cur_data[:, valid_voxels])
        raw_data = np.concatenate(raw_data, axis=-1) # concatenate left and right hemisphere together
        self.num_voxels = raw_data.shape[-1]
        proto_sim = pickle.load(open(os.path.join(proto_dir, subj_name+'.pkl'), 'rb')) # (N, n, num_prototypes)

        # get the index mapping between proto_sim and raw_data
        img_name = proto_sim['image_name'] # (N,)
        img_name = [int(name.split('_')[0].split('-')[1])-1 for name in img_name]
        raw2proto_idx = {cur:idx for idx, cur in enumerate(img_name)}
    
        # generate train and test split
        train_idx, test_idx = train_test_split(img_name, test_size=0.2, random_state=seed)
        
        if split == 'train':
            self.raw_data = raw_data[train_idx] # (N_train, num_voxels)
            self.proto_sim = proto_sim['prototype_similarity'][[raw2proto_idx[idx] for idx in train_idx]] # (N_train, n, num_prototypes)
        else:
            self.raw_data = raw_data[test_idx] # (N_test, num_voxels)
            self.proto_sim = proto_sim['prototype_similarity'][[raw2proto_idx[idx] for idx in test_idx]] # (N_test, n, num_prototypes)

        print('%s voxels selected and %d samples for %s split' %(self.num_voxels, len(self.raw_data), split))

    def __getitem__(self, index):
        return self.proto_sim[index], self.raw_data[index]

    def __len__(self):
        return len(self.raw_data)

class ImageNet_Feature(torch.utils.data.Dataset):
    """
    Dataset for training pixel readout on ImageNet data.
    """
    def __init__(self, feature_dir, split='train', sampled_img_num=None):
        self.feature_dir = feature_dir
        self.split = split

        # gather all image files for training
        self.all_img = []
        self.all_img_cat = []
        cls = os.listdir(feature_dir)
        for c in cls:
            cur_pool = glob(os.path.join(feature_dir, c, '*.npy'))
            if split != 'all':
                if split == 'train':
                    cur_pool = [p for p in cur_pool if not 'val_' in p]
                else:
                    cur_pool = [p for p in cur_pool if 'val_' in p]
                    self.split = 'val'

                if split == 'train' and sampled_img_num is not None:
                    cur_pool = np.random.choice(cur_pool, sampled_img_num, replace=False).tolist()
            else:
                cur_pool = [p for p in cur_pool]   
                if sampled_img_num is not None:
                    cur_pool = np.random.choice(cur_pool, sampled_img_num, replace=False).tolist()      
            self.all_img.extend(cur_pool)
            self.all_img_cat.extend([c]*len(cur_pool))

        
        print("%d Image in total used for %s" %(len(self.all_img), self.split))

    def __getitem__(self, index):
        cur_feat_path = self.all_img[index]
        cur_cat = self.all_img_cat[index]
        img_name = os.path.basename(cur_feat_path)[:-4]

        # load feature
        img_feat = _load_mmap(img_name+'.npy', os.path.dirname(cur_feat_path))

        return img_feat, cur_cat, img_name 

    def __len__(self):
        return len(self.all_img)