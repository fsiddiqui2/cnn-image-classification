from torch.utils.data import Dataset
from PIL import Image
import os
import glob
import random

class CatDogDataset(Dataset):
    """
    A custom PyTorch Dataset for loading cat and dog images from a specified directory
    and splitting them into train, validation, or test sets.

    The expected file structure is:
    root_dir/
    ├── Cat/
    │   ├── 0.jpg
    │   ├── 1.jpg
    │   └── ...
    └── Dog/
        ├── 0.jpg
        ├── 1.jpg
        └── ...
    """
    
    def __init__(self, root_dir: str, split: str, transform=None, train_ratio: float = 0.7, val_ratio: float = 0.15, random_seed: int = 42):
        """
        Initializes the CatDogDataset.

        Args:
            root_dir (str): The root directory containing 'Cat' and 'Dog' subdirectories.
            split (str): Specifies the data split to return. Must be 'train', 'val', or 'test'.
            transform (torchvision.transforms.Compose, optional): A sequence of transformations
                                                                 to apply to the images. Defaults to None.
            train_ratio (float): The proportion of data to use for the training set (0.0 to 1.0).
                                 Defaults to 0.7.
            val_ratio (float): The proportion of data to use for the validation set (0.0 to 1.0).
                               Defaults to 0.15.
            random_seed (int): Seed for reproducibility of the data split. Defaults to 42.

        Raises:
            ValueError: If 'split' is not 'train', 'val', or 'test'.
            FileNotFoundError: If 'Cat' or 'Dog' directories are not found.
        """
        
        if split not in ["train", "val", "test"]:
            raise ValueError("Split must be 'train', 'val', or 'test'.")
        if not (0.0 <= train_ratio <= 1.0 and 0.0 <= val_ratio <= 1.0 and (train_ratio + val_ratio) <= 1.0):
            raise ValueError("train_ratio and val_ratio must be between 0 and 1, and their sum must be <= 1.")

        self.root_dir = root_dir
        self.split = split
        self.transform = transform
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.random_seed = random_seed

        self.data = []
        self._load_data()
        self._split_data()

    def _load_data(self):
        """
        Loads all image paths and assigns labels (0 for cat, 1 for dog).
        """
        cat_dir = os.path.join(self.root_dir, 'Cat')
        dog_dir = os.path.join(self.root_dir, 'Dog')

        if not os.path.exists(cat_dir):
            raise FileNotFoundError(f"Cat directory not found at: {cat_dir}")
        if not os.path.exists(dog_dir):
            raise FileNotFoundError(f"Dog directory not found at: {dog_dir}")

        # Get all image paths for cats and dogs
        cat_images = glob.glob(os.path.join(cat_dir, '*.jpg')) 
        dog_images = glob.glob(os.path.join(dog_dir, '*.jpg')) 

        # Assign labels: 0 for cat, 1 for dog
        for image_path in cat_images:
            # image = Image.open(img_path).convert('RGB')
            # if self.transform:
            #     image = self.transform(image)
            self.data.append((image_path, 0)) # 0 for Cat
        for image_path in dog_images:
            # image = Image.open(img_path).convert('RGB')
            # if self.transform:
            #     image = self.transform(image)
            if "9041.jpg" in image_path: continue # skip corrupted file
            self.data.append((image_path, 1)) # 1 for Dog

        # Shuffle the entire dataset for random splitting
        random.seed(self.random_seed)
        random.shuffle(self.data)

        if not self.data:
            print(f"Warning: No images found in {self.root_dir}. Please check the directory and file extensions.")

    def _split_data(self):
        """
        Splits the loaded data into train, validation, and test sets based on ratios.
        Sets self.data to the appropriate subset based on self.split.
        """
        total_size = len(self.data)
        train_size = int(total_size * self.train_ratio)
        val_size = int(total_size * self.val_ratio)
        test_size = total_size - train_size - val_size

        if self.split == 'train':
            self.data = self.data[:train_size]
        elif self.split == 'val':
            self.data = self.data[train_size : train_size + val_size]
        elif self.split == 'test':
            self.data = self.data[train_size + val_size : train_size + val_size + test_size]

        print(f"Dataset '{self.split}' split created with {len(self.data)} samples out of {total_size} total, seed {self.random_seed}.")

    def class_distribution(self):
        """
        Returns the percent of postive class in this dataset split
        """
        return sum([label for feat,label in self.data]) / len(self.data)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        """
        Retrieves an image and its label from the dataset at the given index.

        Args:
            idx (int): The index of the sample to retrieve.

        Returns:
            tuple: A tuple containing the image (torch.Tensor) and its label (int).
        """
        img_path, label = self.data[idx]
        image = Image.open(img_path).convert('RGB') # Ensure image is in RGB format
        if self.transform:
            image = self.transform(image)

        #return self.data[idx]
        return image, label