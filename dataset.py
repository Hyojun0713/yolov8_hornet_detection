import os
import cv2
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
import config

class CustomDataset(Dataset):
    def __init__(self, img_dirs, transform=None, small_image_augmentation=config.SMALL_IMAGE_AUGMENTATION):
        self.img_paths = []
        self.small_img_paths = []
        for img_dir in img_dirs:
            if 'v5i' in img_dir:  # 작은 이미지 경로 식별
                self.small_img_paths.extend(
                    [os.path.join(img_dir, f) for f in os.listdir(img_dir) if os.path.isfile(os.path.join(img_dir, f))])
            else:
                self.img_paths.extend(
                    [os.path.join(img_dir, f) for f in os.listdir(img_dir) if os.path.isfile(os.path.join(img_dir, f))])

        # 작은 이미지 증강
        self.img_paths += self.small_img_paths * small_image_augmentation
        self.transform = transform

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.transform:
            image = self.transform(image=image)['image']

        return image, img_path

# Albumentations 변환 정의
transform = A.Compose([
    A.RandomResizedCrop(height=640, width=640, scale=(0.3, 1.0)),
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.RandomBrightnessContrast(p=0.2),
    A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=45, p=0.5),
    A.OneOf([
        A.GaussNoise(p=0.5),
        A.MultiplicativeNoise(p=0.5),
    ], p=0.2),
    A.OneOf([
        A.MotionBlur(p=0.2),
        A.MedianBlur(blur_limit=3, p=0.1),
        A.Blur(blur_limit=3, p=0.1),
    ], p=0.2),
    A.OneOf([
        A.OpticalDistortion(p=0.3),
        A.GridDistortion(p=0.1),
        A.ElasticTransform(p=0.3, alpha=120, sigma=120 * 0.05),
    ], p=0.2),
    A.OneOf([
        A.CLAHE(clip_limit=2),
        A.Sharpen(),
        A.Emboss(),
        A.RandomBrightnessContrast(),
    ], p=0.3),
    A.HueSaturationValue(p=0.3),
    ToTensorV2(),
])

def create_dataloader(img_dirs, batch_size=config.BATCH_SIZE):
    dataset = CustomDataset(img_dirs=img_dirs, transform=transform)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)