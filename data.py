from os import listdir
from os.path import join

from torch.utils.data import Dataset
import torchvision.transforms as transforms
from PIL import Image

def is_image_file(filename):
    # Check if the file has an image extension
    return any(filename.endswith(extension) for extension in [".png", ".jpg", ".jpeg", ".bmp"])

def load_img(filepath):
    # Load the image and convert it to YCbCr color space
    img = Image.open(filepath).convert('YCbCr')
    y, cb, cr = img.split()
    return y, cb, cr

CROP_SIZE = 128  # Adjust this size based on your dataset and model requirements

class DatasetFromFolder(Dataset):
    def __init__(self, image_dir, zoom_factor, input_transform=None, target_transform=None):
        super(DatasetFromFolder, self).__init__()
        self.image_filenames = [join(image_dir, x) for x in listdir(image_dir) if is_image_file(x)]
        self.zoom_factor = zoom_factor
        self.input_transform = input_transform
        self.target_transform = target_transform
        self.cbcr_transform = transforms.Compose([
            transforms.Resize((CROP_SIZE, CROP_SIZE)),  # Ensure Cb and Cr are the same size
            transforms.ToTensor()
        ])

    def __getitem__(self, index):
        y, cb, cr = load_img(self.image_filenames[index])
        target_y = y.copy()
        
        # Resize Y channel to fixed size
        y = y.resize((CROP_SIZE, CROP_SIZE), Image.BICUBIC)
        target_y = target_y.resize((CROP_SIZE, CROP_SIZE), Image.BICUBIC)
        
        # Apply input and target transformations
        input_y = self.input_transform(y)
        target_y = self.target_transform(target_y)
        
        # Transform Cb and Cr channels to tensors
        cb = self.cbcr_transform(cb)
        cr = self.cbcr_transform(cr)

        return input_y, target_y, cb, cr

    def __len__(self):
        return len(self.image_filenames)
