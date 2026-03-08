import os
from PIL import Image
from torch.utils.data import Dataset


class FlowerTypeDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform

        self.image_paths = []
        self.labels = []
        self.classes = []

        # get folder names (sites)
        self.classes = sorted(os.listdir(root_dir))
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}

        for cls in self.classes:
            class_folder = os.path.join(root_dir, cls)

            if not os.path.isdir(class_folder):
                continue

            for img in os.listdir(class_folder):
                if img.lower().endswith((".jpg", ".jpeg", ".png")):
                    self.image_paths.append(os.path.join(class_folder, img))
                    self.labels.append(self.class_to_idx[cls])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]

        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, label