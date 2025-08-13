import cv2
import glob
import matplotlib
import numpy as np
import os
import torch

from depth_anything_v2.dpt import DepthAnythingV2


if __name__ == '__main__':
    # Static values for image folder, output directory, and other arguments
    image_folder = 'C:/Users/33849/f/picture'  # Set path to your image folder
    outdir = 'C:/Users/33849/f/output'  # Set path for output images
    input_size = 518  # Set input size for the model
    encoder = 'vitl'  # Choose the encoder ('vitl', 'vitb', 'vits', 'vitg')
    grayscale = False  # Set to True if you want grayscale depth map
    pred_only = False  # Set to True if you want to display only the prediction (depth map)

    DEVICE = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
    
    # Model configurations based on encoder type
    model_configs = {
        'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
        'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
        'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
        'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
    }
    
    # Load the Depth Anything V2 model
    depth_anything = DepthAnythingV2(**model_configs[encoder])
    depth_anything.load_state_dict(torch.load(f'checkpoints/depth_anything_v2_{encoder}.pth', map_location='cpu'))
    depth_anything = depth_anything.to(DEVICE).eval()

    # Get all image files in the folder (any file with an image extension)
    filenames = glob.glob(os.path.join(image_folder, '*.*'))  # Get all files in the folder
    
    # Create output directory if it doesn't exist
    os.makedirs(outdir, exist_ok=True)
    
    margin_width = 50
    cmap = matplotlib.colormaps.get_cmap('Spectral_r')
    
    # Scaling factor for resizing the depth map
    scale_factor = 2  # Adjust this value to make the display box bigger
    
    for k, filename in enumerate(filenames):
        print(f'Processing {k+1}/{len(filenames)}: {filename}')
        
        # Read the image
        raw_image = cv2.imread(filename)
        frame_height, frame_width = raw_image.shape[:2]
        
        # Generate depth map using the model
        depth = depth_anything.infer_image(raw_image, input_size)
        
        # Normalize depth map
        depth = (depth - depth.min()) / (depth.max() - depth.min()) * 255.0
        depth = depth.astype(np.uint8)
        
        # Resize depth map to make it larger
        depth_resized = cv2.resize(depth, (frame_width * scale_factor, frame_height * scale_factor))
        
        # Apply grayscale or color palette
        if grayscale:
            depth_resized = np.repeat(depth_resized[..., np.newaxis], 3, axis=-1)
        else:
            depth_resized = (cmap(depth_resized)[:, :, :3] * 255)[:, :, ::-1].astype(np.uint8)
        
        # Resize the split region (margin)
        split_region_resized = np.ones((frame_height * scale_factor, margin_width * scale_factor, 3), dtype=np.uint8) * 255
        
        # Combine the original frame and resized depth map
        if pred_only:
            output_image = depth_resized
        else:
            combined_image = cv2.hconcat([raw_image, split_region_resized, depth_resized])
            output_image = combined_image
        
        # Save the output image
        output_filename = os.path.join(outdir, os.path.splitext(os.path.basename(filename))[0] + '_depth.png')
        cv2.imwrite(output_filename, output_image)

    print(f'Processing completed. Check the output in {outdir}')
