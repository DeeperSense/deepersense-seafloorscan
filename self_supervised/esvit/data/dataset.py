import os
import glob

from PIL import Image
from torch.utils.data import Dataset


class SonarDataset(Dataset):
    """Loads and stores (transformed) sonar waterfalls.
    Compatible with PyTorch DataLoader and Samplers.
    """

    def __init__(self, folder_path, transform=None):
        super().__init__()
        self.transform = transform
        self.img_files = glob.glob(os.path.join(folder_path, '*.tiff'))

    def __getitem__(self, index):
        sample = Image.open(self.img_files[index]).convert('L')
        if self.transform:
            sample = self.transform(sample)
        return sample

    def __len__(self):
        return len(self.img_files)
