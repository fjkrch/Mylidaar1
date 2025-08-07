import argparse
import cv2
import matplotlib
import numpy as np
import torch
from depth_anything_v2.dpt import DepthAnythingV2

if __name__ == '__main__':
    # Argument parser for customizable options
    parser = argparse.ArgumentParser(description='Depth Anything V2 Real-Time')
    parser.add_argument('--input-size', type=int, default=518, help='Input size for the model')
    parser.add_argument('--encoder', type=str, default='vitl', choices=['vits', 'vitb', 'vitl', 'vitg'], help='Choice of encoder model')
    parser.add_argument('--grayscale', dest='grayscale', action='store_true', help='Do not apply colorful palette')
    parser.add_argument('--pred-only', dest='pred_only', action='store_true', help='Only display the prediction')
    args = parser.parse_args()

    # Select device based on available hardware
    DEVICE = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'

    # Configurations for different encoder models
    model_configs = {
        'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
        'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
        'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
        'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
    }

    # Load Depth Anything V2 model
    depth_anything = DepthAnythingV2(**model_configs[args.encoder])
    depth_anything.load_state_dict(torch.load(f'checkpoints/depth_anything_v2_{args.encoder}.pth', map_location='cpu'))
    depth_anything = depth_anything.to(DEVICE).eval()

    # Initialize webcam and face detection
    cap = cv2.VideoCapture(0)  # Open default webcam
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    # Set your camera's focal length (in pixels)
    focal_length_px = 800  # Adjust this for your camera
    real_face_width_cm = 13  # Average human face width in cm

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Convert the frame to grayscale for face detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        # Generate the depth map using Depth Anything V2
        depth = depth_anything.infer_image(frame, args.input_size)

        # Normalize the depth map for visualization
        depth_vis = (depth - depth.min()) / (depth.max() - depth.min()) * 255.0
        depth_vis = depth_vis.astype(np.uint8)

        # Apply grayscale or colored depth visualization
        if args.grayscale:
            depth_vis = np.repeat(depth_vis[..., np.newaxis], 3, axis=-1)
        else:
            cmap = matplotlib.colormaps.get_cmap('Spectral_r')
            depth_vis = (cmap(depth_vis)[:, :, :3] * 255)[:, :, ::-1].astype(np.uint8)

        # Detect faces and calculate distance
        for (x, y, w, h) in faces:
            center_x = x + w // 2
            center_y = y + h // 2

            if center_y < depth.shape[0] and center_x < depth.shape[1]:
                face_depth = depth[center_y, center_x]
                estimated_distance_cm = (real_face_width_cm * focal_length_px) / w  # Distance estimation from camera

                # Draw face rectangle and display depth information
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(frame, f"Est. Dist: {estimated_distance_cm:.1f}cm", (x, y - 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
                cv2.putText(frame, f"Depth: {face_depth:.2f}", (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # Combine the original frame and depth visualization for side-by-side display
        margin_width = 50
        split_region = np.ones((frame.shape[0], margin_width, 3), dtype=np.uint8) * 255
        display_frame = cv2.hconcat([frame, split_region, depth_vis])

        # Show the frame with depth information
        cv2.imshow('Depth Anything V2 - Real Time', display_frame)

        # Press 'q' to exit the loop
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release resources and close windows
    cap.release()
    cv2.destroyAllWindows()
