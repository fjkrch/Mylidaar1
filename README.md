# 📏 Face Distance Estimation (Pinhole Model + Depth-AI)

This project estimates the distance of a human face (or any object of known size) from a **single camera** using a hybrid approach:  
- **Pinhole Camera Model** (classical geometry)  
- **Depth Anything v2** (AI depth estimation)  

The two signals are combined for more robust and accurate distance estimation.

---

## ✨ Features
- **Face Detection** with OpenCV Haar Cascade  
- **Depth Estimation** using pre-trained Depth Anything v2  
- **Pinhole Camera Model** for geometric distance calculation  
- **Fusion** of classical geometry + AI depth  
- Works with **webcam (real-time)** or **video input**  
- Supports **logging results** to CSV/JSON  
- Can scale on **any object with known size**, not just faces  

---

## 📚 How It Works

1. Detect face/object → get bounding box `(x, y, w, h)`  
2. Estimate distance with **pinhole model**:  
   ```
   Z_pin = (f * L) / l_px
   ```
3. Estimate relative depth with **Depth Anything v2**  
4. Scale and fuse:  
   ```
   Z = α * Z_pin + (1 - α) * Z_ai
   ```

---

## ⚙️ Installation

```bash

# (Optional) Create virtual environment
python -m venv .venv
source .venv/bin/activate   # macOS/Linux
# .venv\Scripts\Activate.ps1 # Windows PowerShell

# Install dependencies
pip install -r requirements.txt

# Create folder for checkpoints
mkdir -p checkpoints

# Download Depth Anything v2 weights into ./checkpoints
# Example: checkpoints/depth_anything_v2.pth
```

---

## 🚀 How to Use

### Run with Webcam
```bash
python src/run_video.py
```
- Opens webcam (`--camera 0` default)  
- Press **s** → save point cloud  
- Press **q** → quit  

### Run with Video File
```bash
python src/run_video.py --video path/to/input.mp4
```

### Visualize Saved Point Cloud
```bash
python src/run_pointcloud.py
```


