# Silkscan: Quick Start Tutorial for Beginners

Welcome to the Silkscan pipeline! This tutorial is designed for users who want to quickly process a video file and convert it into a 3D point cloud without writing any code.

We've provided a simple command-line tool, `process_video.py`, which handles all the complicated settings for you.

---

## 1. Prerequisites

Before you begin, ensure you have the following ready:
1.  **Terminal Access:** You need to open your terminal (Terminal on Mac/Linux, or Command Prompt / PowerShell on Windows).
2.  **A Video File:** Ensure you have the video file you want to process (e.g., `my_scan.mp4`) located somewhere on your computer.
3.  **Correct Folder:** Ensure you are in the `silkscan` project directory in your terminal. You can navigate there using the `cd` command:
    ```bash
    cd path/to/silkscan
    ```

---

## 2. Processing Your Video

The `process_video.py` script requires only one thing: the path to your video file.

### The Basic Command
To run the script with the default, highly-optimized settings, simply type:

```bash
python process_video.py path/to/your/video.mp4
```

*Replace `path/to/your/video.mp4` with the actual location of your video file.*

### Changing the Output Name
By default, the script saves your 3D model as `output.pcd` in the same folder. If you want to give it a specific name, use the `-o` (or `--output`) option:

```bash
python process_video.py path/to/your/video.mp4 -o my_3d_model.pcd
```

### Advanced Options (Optional)
If you know the physical scale of your camera setup, you can adjust these settings. Otherwise, it is safe to ignore them.
*   `--mm-per-frame`: How many millimeters the camera moves between each video frame (Default: 0.1).
*   `--pixels-per-mm`: The scale of pixels to millimeters (Default: 1.0).

**Example:**
```bash
python process_video.py my_scan.mp4 -o my_3d_model.pcd --mm-per-frame 0.2
```

---

## 3. Viewing Your 3D Model

Once the script finishes processing, you will have a `.pcd` (Point Cloud Data) file. 

You cannot open this file with a standard photo viewer. You need specialized software. We recommend **MeshLab** or **CloudCompare**, which are both free and easy to use.

### Viewing with MeshLab
1. Download and install [MeshLab](https://www.meshlab.net/).
2. Open MeshLab.
3. Go to `File` -> `Import Mesh...` (or just drag and drop the `.pcd` file into the window).
4. You can now use your mouse to click and drag to rotate the 3D model!

### Viewing with Python (Open3D)
If you have Open3D installed, you can quickly view it from the command line by running:
```bash
python -c "import open3d as o3d; pcd = o3d.io.read_point_cloud('output.pcd'); o3d.visualization.draw_geometries([pcd])"
```

---

## 4. Troubleshooting

*   **Error: "The video file was not found."**
    Make sure you spelled the path to the video file correctly. If the video is in another folder, you must provide the full path (e.g., `/Users/name/Desktop/video.mp4`).

*   **Error: "ModuleNotFoundError: No module named 'cv2'"** (or similar)
    This means your Python environment doesn't have the required packages installed. Ensure you have installed the requirements, typically by running:
    ```bash
    pip install -r prototype/requirements.txt
    ```

*   **The point cloud looks stretched or flat.**
    This usually means the `--mm-per-frame` parameter is incorrect for your specific video. Try adjusting this number higher or lower when you run the script.
