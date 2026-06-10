# Silkscan: Quick Start Tutorial for Beginners

Welcome to the Silkscan pipeline! This tutorial is designed for users who want to quickly process a video file and convert it into a 3D point cloud without writing any code.

We've provided a simple Graphical User Interface (GUI), `silkscan_gui.py`, which handles all the complicated settings, cropping, and visualization for you.

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

## 2. Launching the Interface

To open the graphical interface, run the following command in your terminal:

```bash
python silkscan_gui.py
```

A window titled **"Silkscan 3D Reconstruction"** will pop up.

---

## 3. Processing Your Video

1.  **Select Video:** Click the **"Browse"** button next to the *Video File* field and select your `.mp4` scan.
    *   *Smart Scaling:* If you have a `manifest.json` file in the same folder as your video (which defines the physical scale of your camera setup), the GUI will automatically detect it and fill in the correct **Scaling Parameters** for you!
2.  **Output Location:** The GUI will automatically suggest saving the 3D model as a `.pcd` file next to your video. You can click "Browse" to change the location or name.
3.  **Options:**
    *   **Enable Auto-Cropping:** Leave this checked! The tool will automatically detect the borders of the capture box and mask out background noise before processing.
    *   **Auto-Crop Padding (px):** Adjust this if the background mask is too tight or too loose around your spider web. Default is 50.
    *   **Extraction Method:** 
        *   `Threshold`: A fast method that extracts based on pixel brightness. The tool will automatically extract a range of thresholds (`0.15` to `0.90`) for you to compare!
        *   `Steger`: A highly-accurate but slower line-detection method. NOTE: this is experimental, may not be as consistent as the other method.
4.  **Run:** Click **"Start Processing & Open Editor"**. 

*Note: Processing a full video can take several minutes. You can monitor the progress directly in the black console window at the bottom of the GUI!*

---

## 4. Cropping and Saving Your 3D Model

When processing finishes, the Tkinter window will close and an interactive **Silkscan Crop Editor** will automatically open.

1. **Adjust Boundaries:** Use the sliders on the left (X, Y, and Z Min/Max) to trim off any remaining pieces of the capture box walls. The 3D model will instantly update as you drag!
2. **Save Results:** Once your spider web is perfectly isolated, click **"Save All Thresholds"** (or "Save Cropped PCD" if using Steger). The editor will apply your exact bounding box crop and save your final 3D point cloud files to your selected output folder.

If you want to view the `.pcd` file later, you can use specialized 3D software like **MeshLab** or **CloudCompare** (both are free).
1. Download and install [MeshLab](https://www.meshlab.net/).
2. Open MeshLab.
3. Go to `File` -> `Import Mesh...` and select your `.pcd` file.

---

## 5. Troubleshooting

**"The program runs until the stacking is complete, then crashes!"**
This is a known bug with `open3d` on certain macOS environments where the 3D graphics engine (Filament) fails to initialize and throws a segmentation fault (SIGSEGV).
**The Fix:** We've implemented an automatic safety-net! If the modern editor crashes on your machine, the script will instantly save a raw backup of your 3D model and automatically launch a **Robust Legacy Editor Mode**. This legacy editor uses an older, crash-proof graphics engine, allowing you to seamlessly continue adjusting your boundaries and saving your files without losing any data!

*   **Error: "ModuleNotFoundError: No module named 'tkinter'"**
    Tkinter is usually included with Python, but if you get this error, you may need to install it via your package manager (e.g., `brew install python-tk` on Mac).
*   **The point cloud looks stretched or flat.**
    This means your scaling parameters (`mm / frame` or `pixels / mm`) are incorrect. If you didn't have a `manifest.json`, you will need to manually adjust these values in the GUI based on your physical camera setup.
