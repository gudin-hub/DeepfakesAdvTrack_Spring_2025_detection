import os
import cv2

from PIL import Image
from torch.utils.data import dataset

class FolderDataset(dataset.Dataset):
    def __init__(self, folder, transforms):
        self.folder = folder
        self.img_list = open(os.path.join(folder, "img_list.txt"), "r").readlines()
        self.face_info = open(os.path.join(folder, "face_info.txt"), "r").readlines()

        assert len(self.img_list) == len(self.face_info)

        self.transforms = transforms

    def __len__(self):
        return len(self.img_list)
    
    def get_img_name(self):
        return self.img_list

    def read_crop_face(self, idx, scale = 1.3):
        img = cv2.imread(os.path.join(self.folder, "imgs", self.img_list[idx].strip()))
        height, width = img.shape[:2]

        box = self.face_info[idx].split(" ")
        box = [float(x) for x in box]
        x1 = box[0]
        y1 = box[1]
        x2 = box[2]
        y2 = box[3]
        center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2

        # enlarge the bbox by 1.3 and crop
        size_bb = int(max(x2 - x1, y2 - y1) * scale)
        x1 = max(int(center_x - size_bb // 2), 0) # Check for out of bounds, x-y top left corner
        y1 = max(int(center_y - size_bb // 2), 0)
        size_bb = min(width - x1, size_bb)
        size_bb = min(height - y1, size_bb)

        cropped_face = img[y1:y1 + size_bb, x1:x1 + size_bb]
        return cropped_face

    def __getitem__(self, idx):
        img = self.read_crop_face(idx)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img)
        img = self.transforms(img)
        return img