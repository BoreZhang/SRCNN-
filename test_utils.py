import unittest
import os
import shutil
from PIL import Image
import torch
from torchvision import transforms
from data import is_image_file, load_img, DatasetFromFolder

class TestDataMethods(unittest.TestCase):
    def setUp(self):
        if os.path.exists('test_images'):
            shutil.rmtree('test_images')
        os.makedirs('test_images', exist_ok=True)

        img = Image.new('RGB', (100, 100), color='red')
        img.save('test_images/test1.jpg')
        img.save('test_images/test2.png')
        img.save('test_images/test3.bmp')
        img.save('test_images/test4.gif')  # This should not be counted as an image file by is_image_file

    def tearDown(self):
        shutil.rmtree('test_images')

    def test_is_image_file(self):
        self.assertTrue(is_image_file('test1.jpg'))
        self.assertTrue(is_image_file('test2.png'))
        self.assertTrue(is_image_file('test3.bmp'))
        self.assertFalse(is_image_file('test4.gif'))

    def test_load_img(self):
        y, cb, cr = load_img('test_images/test1.jpg')
        self.assertEqual(y.mode, 'L')  # 'L' mode for grayscale image
        self.assertEqual(cb.mode, 'L')
        self.assertEqual(cr.mode, 'L')

    def test_DatasetFromFolder(self):
        input_transform = transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.ToTensor()
        ])
        target_transform = transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.ToTensor()
        ])
        dataset = DatasetFromFolder('test_images', zoom_factor=2, input_transform=input_transform, target_transform=target_transform)
        
        self.assertEqual(len(dataset), 3)  # Only .jpg, .png, and .bmp files should be counted

        sample, target, cb, cr = dataset[0]
        self.assertEqual(sample.size(), torch.Size([1, 128, 128]))
        self.assertEqual(target.size(), torch.Size([1, 128, 128]))
        self.assertEqual(cb.size(), torch.Size([1, 128, 128]))
        self.assertEqual(cr.size(), torch.Size([1, 128, 128]))

if __name__ == '__main__':
    unittest.main()
