import torch
import numpy as np
import cv2
from torch.utils.data import Dataset
import prepare_data
from albumentations.torch.functional import img_to_tensor


img_size = 512

class RoboticsDataset(Dataset):
    def __init__(self, file_names, to_augment=False, transform=None, mode='train', problem_type=None):
        self.file_names = file_names
        self.to_augment = to_augment
        self.transform = transform
        self.mode = mode
        self.problem_type = problem_type

    def __len__(self):
        return len(self.file_names)

    def __getitem__(self, idx):
        img_file_name = self.file_names[idx]
        image = load_image(img_file_name)
        # image = image.transpose(0,2,1)
        mask = load_mask(img_file_name, self.problem_type)

        data = {"image": image, "mask": mask}
        augmented = self.transform(**data)
        image, mask = augmented["image"], augmented["mask"]

        if self.mode == 'train':
            if self.problem_type == 'binary':
                return img_to_tensor(image), torch.from_numpy(np.expand_dims(mask, 0)).float()
            else:
                return img_to_tensor(image), torch.from_numpy(mask).long()
        else:
            return img_to_tensor(image), str(img_file_name)


def load_image(path):
    img = cv2.imread(str(path))
    img = cv2.resize(img, (img_size, img_size), interpolation=cv2.INTER_CUBIC)
    #print(img.shape)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    #img = img.transpose(0,2,1)
    #print(img.shape)
    #return img


def load_mask(path, problem_type):
    if problem_type == 'binary':
        mask_folder = 'label'
        factor = prepare_data.binary_factor
    elif problem_type == 'parts':
        mask_folder = 'parts_masks'
        factor = prepare_data.parts_factor
    elif problem_type == 'instruments':
        factor = prepare_data.instrument_factor
        mask_folder = 'instruments_masks'


    #print(str(path).replace('image', mask_folder).replace('jpg', 'png'))

    mask = cv2.imread(str(path).replace('image', mask_folder).replace('jpg', 'png'), 0)
    # mask = cv2.imread(str(path).replace('image', mask_folder).replace('im', 'gt'), 0)
    mask = cv2.resize(mask, (img_size, img_size), interpolation=cv2.INTER_CUBIC) 

    return (mask / factor).astype(np.uint8)
