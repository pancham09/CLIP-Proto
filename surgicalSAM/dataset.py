from torch.utils.data import Dataset
import os 
import os.path as osp
import re 
import numpy as np 
import cv2 

class Endovis18Dataset(Dataset):
    def __init__(self, data_root_dir = "../data/endovis_2018", 
                 mode = "val", 
                 vit_mode = "h",
                 version = 0):
        
        """Define the Endovis18 dataset with Temporal Pairing

        Args:
            data_root_dir (str, optional): root dir containing all data for Endovis18. Defaults to "../data/endovis_2018".
            mode (str, optional): either in "train" or "val" mode. Defaults to "val".
            vit_mode (str, optional): "h", "l", "b" for huge, large, and base versions of SAM. Defaults to "h".
            version (int, optional): augmentation version to use. Defaults to 0.
        """
        
        self.vit_mode = vit_mode
       
        # directory containing all binary annotations
        if mode == "train":
            self.mask_dir = osp.join(data_root_dir, mode, str(version), "binary_annotations")
        elif mode == "val":
            self.mask_dir = osp.join(data_root_dir, mode, "binary_annotations")

        self.temporal_pairs = []
        
        # Iterate through sequences to prevent cross-sequence bleeding
        for subdir, _, files in os.walk(self.mask_dir):
            if len(files) == 0:
                continue 
            
            # 1. Group all files in this specific sequence by their Class ID
            class_dict = {}
            for f in files:
                cls_id = int(re.search(r"class(\d+)", f).group(1))
                if cls_id not in class_dict:
                    class_dict[cls_id] = []
                class_dict[cls_id].append(f)
                
            # 2. Sort chronologically and build pairs within the same class and sequence
            for cls_id, c_files in class_dict.items():
                c_files = sorted(c_files) # Ensures chronological order (e.g., frame000, frame001)
                
                # Pair consecutive frames
                for i in range(len(c_files) - 1):
                    mask_name_1 = osp.join(osp.basename(subdir), c_files[i])
                    mask_name_2 = osp.join(osp.basename(subdir), c_files[i+1])
                    
                    self.temporal_pairs.append((mask_name_1, mask_name_2))

        # Assign the pre-computed pairs to the mask list
        self.mask_list = self.temporal_pairs

    def __len__(self):
        return len(self.mask_list)

    def __getitem__(self, index):
        # Unpack the pre-validated temporal pair
        mask_name_1, mask_name_2 = self.mask_list[index]
        
        # Get class ID (identical for both masks in the pair)
        cls_id = int(re.search(r"class(\d+)", mask_name_1).group(1))
        
        # Get pre-computed SAM feature for Frame 1
        feat_dir_1 = osp.join(self.mask_dir.replace("binary_annotations", f"sam_features_{self.vit_mode}"), mask_name_1.split("_")[0] + ".npy")
        sam_feat_1 = np.load(feat_dir_1)
        
        # Get pre-computed SAM feature for Frame 2
        feat_dir_2 = osp.join(self.mask_dir.replace("binary_annotations", f"sam_features_{self.vit_mode}"), mask_name_2.split("_")[0] + ".npy")
        sam_feat_2 = np.load(feat_dir_2)
        
        # Get ground-truth masks for both frames
        mask_1 = cv2.imread(osp.join(self.mask_dir, mask_name_1), cv2.IMREAD_GRAYSCALE)
        mask_2 = cv2.imread(osp.join(self.mask_dir, mask_name_2), cv2.IMREAD_GRAYSCALE)
        
        # Get static class embedding
        class_embedding_path = osp.join(self.mask_dir.replace("binary_annotations", f"class_embeddings_{self.vit_mode}"), mask_name_1.replace("png","npy"))
        class_embedding = np.load(class_embedding_path)

        return sam_feat_1, sam_feat_2, mask_name_1, mask_name_2, cls_id, mask_1, mask_2, class_embedding
 

class Endovis17Dataset(Dataset):
    def __init__(self, data_root_dir = "../data/endovis_2017", 
                 mode = "val",
                 fold = 0,  
                 vit_mode = "h",
                 version = 0):
                        
        self.vit_mode = vit_mode
        
        all_folds = list(range(1, 9))
        fold_seq = {0: [1, 3],
                    1: [2, 5],
                    2: [4, 8],
                    3: [6, 7]}
        
        if mode == "train":
            seqs = [x for x in all_folds if x not in fold_seq[fold]]     
        elif mode == "val":
            seqs = fold_seq[fold]

        self.mask_dir = osp.join(data_root_dir, str(version), "binary_annotations")
        
        self.temporal_pairs = []
        
        # Iterate through specific folds/sequences
        for seq in seqs:
            seq_path = osp.join(self.mask_dir, f"seq{seq}")
            if not os.path.exists(seq_path):
                continue
                
            files = os.listdir(seq_path)
            if len(files) == 0:
                continue
                
            # 1. Group files by Class ID
            class_dict = {}
            for f in files:
                cls_id = int(re.search(r"class(\d+)", f).group(1))
                if cls_id not in class_dict:
                    class_dict[cls_id] = []
                class_dict[cls_id].append(f)
                
            # 2. Sort chronologically and build pairs
            for cls_id, c_files in class_dict.items():
                c_files = sorted(c_files)
                
                for i in range(len(c_files) - 1):
                    mask_name_1 = f"seq{seq}/{c_files[i]}"
                    mask_name_2 = f"seq{seq}/{c_files[i+1]}"
                    
                    self.temporal_pairs.append((mask_name_1, mask_name_2))
            
        self.mask_list = self.temporal_pairs
            
    def __len__(self):
        return len(self.mask_list)

    def __getitem__(self, index):
        # Unpack the pre-validated temporal pair
        mask_name_1, mask_name_2 = self.mask_list[index]
        
        # Get class ID
        cls_id = int(re.search(r"class(\d+)", mask_name_1).group(1))
        
        # Get SAM features for Frame 1
        feat_dir_1 = osp.join(self.mask_dir.replace("binary_annotations", f"sam_features_{self.vit_mode}"), mask_name_1.split("_")[0] + ".npy")
        sam_feat_1 = np.load(feat_dir_1)
        
        # Get SAM features for Frame 2
        feat_dir_2 = osp.join(self.mask_dir.replace("binary_annotations", f"sam_features_{self.vit_mode}"), mask_name_2.split("_")[0] + ".npy")
        sam_feat_2 = np.load(feat_dir_2)
        
        # Get ground-truth masks
        mask_1 = cv2.imread(osp.join(self.mask_dir, mask_name_1), cv2.IMREAD_GRAYSCALE)
        mask_2 = cv2.imread(osp.join(self.mask_dir, mask_name_2), cv2.IMREAD_GRAYSCALE)
        
        # Get class embedding
        class_embedding_path = osp.join(self.mask_dir.replace("binary_annotations", f"class_embeddings_{self.vit_mode}"), mask_name_1.replace("png","npy"))
        class_embedding = np.load(class_embedding_path)
        
        return sam_feat_1, sam_feat_2, mask_name_1, mask_name_2, cls_id, mask_1, mask_2, class_embedding