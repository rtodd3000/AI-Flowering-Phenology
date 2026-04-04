import os
from PIL import Image
from torch.utils.data import Dataset


class FlowerTypeDataset(Dataset):
    def __init__(self, df, root_dir, transform=None, label_col="flower_type"):
        self.df        = df.reset_index(drop=True)
        self.root_dir  = root_dir
        self.transform = transform
        self.label_col = label_col

        self.classes      = sorted(self.df[label_col].unique().tolist())
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row      = self.df.iloc[idx]
        img_path = os.path.join(self.root_dir, row["site"], row["image_name"])

        # Fallback: try adding .jpg if file not found
        if not os.path.isfile(img_path):
            jpg_path = img_path + ".jpg"
            if os.path.isfile(jpg_path):
                img_path = jpg_path
            else:
                raise FileNotFoundError(
                    f"Image not found: {img_path} (also tried {jpg_path})"
                )

        image = Image.open(img_path).convert("RGB")
        label = self.class_to_idx[row[self.label_col]]

        if self.transform:
            image = self.transform(image)

        return image, label