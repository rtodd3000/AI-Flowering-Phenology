import os
from PIL import Image
from torch.utils.data import Dataset


class FlowerTypeDataset(Dataset):
    def __init__(self, df, root_dir, transform=None):
        self.df = df.reset_index(drop=True)
        self.root_dir = root_dir
        self.transform = transform

        # Classes from dataframe
        self.classes = sorted(self.df["flower_type"].unique())
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}

    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = os.path.join(self.root_dir, row["site"], row["image_name"])

        if not os.path.isfile(img_path):
            img_path = img_path + ".jpg"

        image = Image.open(img_path).convert("RGB")
        label = self.class_to_idx[row["flower_type"]]

        if self.transform:
            image = self.transform(image)

        return image, label