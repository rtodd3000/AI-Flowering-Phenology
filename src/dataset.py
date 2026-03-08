import os
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset

class FlowerDataset(Dataset):
    def __init__(self, csv_file, img_dir, transform=None):
        self.data = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.transform = transform

        # convert labels to numbers
        self.flower_types = {name: i for i, name in enumerate(self.data['flower_type'].unique())}

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]

        img_path = os.path.join(self.img_dir, row["image_name"])
        image = Image.open(img_path).convert("RGB")

        intensity = row["intensity"]
        flower_type = self.flower_types[row["flower_type"]]

        if self.transform:
            image = self.transform(image)

        return image, intensity, flower_type