import argparse
import torch
from model import SRCNN
from data import load_img
from PIL import Image
import torchvision.transforms as transforms

def super_resolve_image(model, image_path, zoom_factor, device):
    model.eval()
    try:
        img = Image.open(image_path).convert('YCbCr')
    except Exception as e:
        print(f"Error loading image: {e}")
        return None
    
    y, cb, cr = img.split()

    input_transform = transforms.Compose([
        transforms.Resize((y.size[1] * zoom_factor, y.size[0] * zoom_factor), Image.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))  # 归一化
    ])
    
    output_transform = transforms.Compose([
        transforms.Normalize((-1,), (2,)),  # 反归一化
        transforms.ToPILImage()
    ])
    
    input_y = input_transform(y).unsqueeze(0).to(device)

    with torch.no_grad():
        output_y = model(input_y).squeeze(0).cpu()

    output_y = output_transform(output_y)
    output_cb = cb.resize(output_y.size, Image.BICUBIC)
    output_cr = cr.resize(output_y.size, Image.BICUBIC)
    output_img = Image.merge('YCbCr', [output_y, output_cb, output_cr]).convert('RGB')

    return output_img

def main():
    parser = argparse.ArgumentParser(description='SRCNN Super Resolution')
    parser.add_argument('--model-path', type=str, required=True, help='Path to the pretrained model')
    parser.add_argument('--image-path', type=str, required=True, help='Path to the input image')
    parser.add_argument('--output-path', type=str, required=True, help='Path to save the output image')
    parser.add_argument('--zoom-factor', type=int, default=2, help='Zoom factor for super resolution')
    parser.add_argument('--cuda', action='store_true', help='Use CUDA if available')
    args = parser.parse_args()

    device = torch.device('cuda' if args.cuda and torch.cuda.is_available() else 'cpu')

    model = SRCNN()
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.to(device)

    output_img = super_resolve_image(model, args.image_path, args.zoom_factor, device)
    if output_img:
        output_img.save(args.output_path)
        print(f'Output image saved to {args.output_path}')
    else:
        print('Failed to generate super-resolved image.')

if __name__ == '__main__':
    main()
