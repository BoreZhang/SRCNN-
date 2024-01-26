from os import listdir
from os.path import join

from torch.utils.data import Dataset
import torchvision.transforms as transforms
from PIL import Image, ImageFilter

def is_image_file(filename):
    # Check if the file has an image extension
    return any(filename.endswith(extension) for extension in [".png", ".jpg", ".jpeg"])

def load_img(filepath):
    # Load the image and convert it to YCbCr color space
    img = Image.open(filepath).convert('YCbCr')
    y, _, _ = img.split()
    return y

CROP_SIZE = 32

class DatasetFromFolder(Dataset):
    def __init__(self, image_dir, zoom_factor):
        super(DatasetFromFolder, self).__init__()
        self.image_filenames = [join(image_dir, x) for x in listdir(image_dir) if is_image_file(x)]

        crop_size = CROP_SIZE - (CROP_SIZE % zoom_factor) # Valid crop size
        self.input_transform = transforms.Compose([transforms.CenterCrop(crop_size), # cropping the image
                                      transforms.Resize(crop_size//zoom_factor),  # subsampling the image (half size)
                                      transforms.Resize(crop_size, interpolation=Image.BICUBIC),  # bicubic upsampling to get back the original size 
                                      transforms.ToTensor()])
        self.target_transform = transforms.Compose([transforms.CenterCrop(crop_size), # since it's the target, we keep its original quality
                                       transforms.ToTensor()])

    def __getitem__(self, index):
        input = load_img(self.image_filenames[index])
        target = input.copy()
        
        # Apply Gaussian blur to the input image
        # input = input.filter(ImageFilter.GaussianBlur(1)) 
        
        # Apply input and target transformations
        input = self.input_transform(input)
        target = self.target_transform(target)

        return input, target

    def __len__(self):
        return len(self.image_filenames)
