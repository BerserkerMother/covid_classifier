import os
import glob
from PIL import Image

from torch.utils import data


class CovidDataset(data.Dataset):
    def __init__(self, root, transform):
        self.root = root
        self.transform = transform

        # empty list of data which contains path to images &
        # list of targets which contains corresponding labels
        self.data = []
        self.target = []

        # read covid images path
        for path in glob.glob(os.path.join(root, 'CT_Covid') + "/*"):
            self.data.append(path)
            self.target.append(1)

        # read non covid images path
        for path in glob.glob(os.path.join(root, 'CT_NonCovid') + "/*"):
            self.data.append(path)
            self.target.append(0)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        path = self.data[idx]
        label = self.target[idx]

        image = Image.open(path).convert(mode="RGB")

        if self.transform:
            image = self.transform(image)

        return image, label
