import unittest
from PIL import Image
import os

import torch
from data import is_image_file, load_img, DatasetFromFolder

class TestDataMethods(unittest.TestCase):

    def test_is_image_file(self):
        # Check if the file extension is an image file
        self.assertTrue(is_image_file('test.png'))
        self.assertTrue(is_image_file('test.jpg'))
        self.assertTrue(is_image_file('test.jpeg'))
        self.assertFalse(is_image_file('test.txt'))

    def test_load_img(self):
        # Create a test image
        img = Image.new('RGB', (60, 30), color = 'red')
        img.save('test_img.jpg')

        # Load the test image and check its properties
        loaded_img = load_img('test_img.jpg')
        self.assertEqual(loaded_img.mode, 'L')  # 'L' mode for grayscale image
        self.assertEqual(loaded_img.size, img.size)

        # Clean up
        os.remove('test_img.jpg')

    def test_DatasetFromFolder(self):
        # Assuming you have a directory named 'test_images' with some images
        # Create a dataset from the folder and check if it contains image filenames
        dataset = DatasetFromFolder('test_images', 2)
        self.assertTrue(len(dataset.image_filenames) > 0)

        # Check if each image filename in the dataset is a valid image file
        for img_file in dataset.image_filenames:
            self.assertTrue(is_image_file(img_file))

        # Check the transformations
        # Create a sample image and apply the input transformation
        sample = Image.new('RGB', (64, 64), color = 'blue')
        transformed_sample = dataset.input_transform(sample)
        self.assertEqual(transformed_sample.size, (32, 32))  # Check the size after transformations

if __name__ == '__main__':
    unittest.main()