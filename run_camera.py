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
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)  # Use DirectShow instead of MSMF  # Open default webcam
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

        # Resize frame for depth model input
        frame_for_depth = cv2.resize(frame, (args.input_size, args.input_size))

        # Generate the depth map using Depth Anything V2
        depth = depth_anything.infer_image(frame_for_depth, args.input_size)

        # Print AI-estimated depth of the pixel in the middle of the frame
        mid_y = depth.shape[0] // 2
        mid_x = depth.shape[1] // 2
        print(f"AI-estimated depth at center pixel: {depth[mid_y, mid_x]:.4f}")

        # Normalize the depth map for visualization
        depth_vis = (depth - depth.min()) / (depth.max() - depth.min()) * 255.0
        depth_vis = depth_vis.astype(np.uint8)

        # Resize the depth map back to original frame size
        depth_resized = cv2.resize(depth_vis, (frame.shape[1], frame.shape[0]))

        # Apply grayscale or colored depth visualization
        if args.grayscale:
            depth_resized = np.repeat(depth_resized[..., np.newaxis], 3, axis=-1)
        else:
            cmap = matplotlib.colormaps.get_cmap('Spectral_r')
            depth_resized = (cmap(depth_resized)[:, :, :3] * 255)[:, :, ::-1].astype(np.uint8)

        # Detect faces and calculate distance
        for (x, y, w, h) in faces:
            center_x = x + w // 2
            center_y = y + h // 2
            depth_for_display = cv2.resize(depth, (frame.shape[1], frame.shape[0]))
            if center_y < depth_for_display.shape[0] and center_x < depth_for_display.shape[1]:
                face_depth = depth_for_display[center_y, center_x]
                estimated_distance_cm = (real_face_width_cm * focal_length_px) / w  # Distance estimation from camera

                # Draw face rectangle and display depth information
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(frame, f"Est. Dist: {estimated_distance_cm:.1f}cm", (x, y - 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
                cv2.putText(frame, f"Depth: {face_depth:.2f}", (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # Prepare a white split region (adjusted for the frame size)
        margin_width = 50
        split_region = np.ones((frame.shape[0], margin_width, 3), dtype=np.uint8) * 255

        # Combine the frame and depth map
        combined_frame = cv2.hconcat([frame, split_region, depth_resized])

        # Show the frame with depth information
        cv2.imshow('Depth Anything V2 - Real Time', combined_frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            # Save AI-estimated depth as point cloud (PLY, XYZ only)
            depth_for_save = cv2.resize(depth, (frame.shape[1], frame.shape[0]))
            h, w = depth_for_save.shape
            xx, yy = np.meshgrid(np.arange(w), np.arange(h))
            points = np.stack([xx.flatten(), yy.flatten(), depth_for_save.flatten()], axis=1)

            # Save original point cloud
            ply_header = '''ply
format ascii 1.0
element vertex {vertex_count}
property float x
property float y
property float z
end_header
'''
            with open('depth_points.ply', 'w') as f:
                f.write(ply_header.format(vertex_count=points.shape[0]))
                np.savetxt(f, points, fmt='%.4f %.4f %.4f')
            print("PLY point cloud saved as depth_points.ply")

            # Find max AI depth and face depth for scaling
            Q1 = np.percentile(depth_for_save, 25)
            Q3 = np.percentile(depth_for_save, 75)
            IQR = Q3 - Q1

            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR

            filtered_depth = depth_for_save[(depth_for_save >= lower_bound) & (depth_for_save <= upper_bound)]
            filtered_depth = np.percentile(filtered_depth,95)  # Use 99th percentile to avoid outliers
            max_ai_depth = np.max(filtered_depth)
            face_depths = []
            for (x, y, w, h) in faces:
                center_x = x + w // 2
                center_y = y + h // 2
                if center_y < depth_for_save.shape[0] and center_x < depth_for_save.shape[1]:
                    face_depths.append(depth_for_save[center_y, center_x])
            if face_depths:
                face_depth = max(face_depths)  # Use max face depth if multiple faces
                pinhole_estimate_cm = (real_face_width_cm * focal_length_px) / w if w != 0 else 0
                # Example: scale AI depth difference to cm using pinhole model
                ai_diff = max_ai_depth - face_depth
                if ai_diff != 0 and pinhole_estimate_cm != 0:
                    scale = pinhole_estimate_cm / ai_diff  # 1 AI depth unit = scale cm
                    # Multiply Z axis by scale after subtracting face_depth
                    scaled_z = (depth_for_save - face_depth) * scale
                    points_scaled = np.stack([xx.flatten(), yy.flatten(), scaled_z.flatten()], axis=1)
                    comment = f"# a1: 1 AI depth unit = {scale:.4f} cm (face depth={face_depth:.2f}, pinhole={pinhole_estimate_cm:.2f}cm, max AI depth={max_ai_depth:.2f})\n"
                    with open('depth_points_scaled.ply', 'w') as f:
                        f.write(comment)
                        f.write(ply_header.format(vertex_count=points_scaled.shape[0]))
                        np.savetxt(f, points_scaled, fmt='%.4f %.4f %.4f')
                    print("Scaled PLY point cloud saved as depth_points_scaled.ply")
       

    # Release resources and close windows
    cap.release()
    cv2.destroyAllWindows()
