# Silkscan: Quick Start Tutorial for Beginners

Welcome to the Silkscan pipeline! This tutorial is designed for users who want to quickly process a video file and convert it into a 3D point cloud without writing any code.

We've provided a simple Graphical User Interface (GUI), `silkscan_gui.py`, which handles all the complicated settings, cropping, and visualization for you.

---

## 1. Prerequisites

Before you begin, ensure you have the following ready:
1.  **Python 3.11:** This project is validated against **Python 3.11**. Check your version with `python3.11 --version`. (If you don't have it, download it from [python.org](https://www.python.org/downloads/) — the official installer includes `tkinter`, which the GUI needs.)
2.  **Terminal Access:** You need to open your terminal (Terminal on Mac/Linux, or Command Prompt / PowerShell on Windows).
3.  **A Video File:** Ensure you have the video file you want to process (e.g., `my_scan.mp4`) located somewhere on your computer.
4.  **Correct Folder:** Ensure you are in the `silkscan` project directory in your terminal. You can navigate there using the `cd` command:
    ```bash
    cd path/to/silkscan
    ```

---

## 2. Setting Up a Virtual Environment (Do This First!)

A **virtual environment** (venv) is a private, self-contained folder that holds the *exact* package versions Silkscan needs, isolated from the rest of your system. **This step is essential:** the most common crash — the program reaching 100% on the stacking phase and then dying with a segmentation fault — is caused by mismatched library versions (especially `numpy` and `open3d`). Installing the pinned `requirements.txt` inside a venv guarantees every machine runs the same validated combination.

You only need to do steps 1 and 3 **once**. After that, you just activate the environment (step 2) each time you open a new terminal.

**1. Create the virtual environment** (creates a `.venv` folder in the project directory):
```bash
python3.11 -m venv .venv
```

**2. Activate it:**

*   On **Mac / Linux**:
    ```bash
    source .venv/bin/activate
    ```
*   On **Windows** (PowerShell):
    ```powershell
    .venv\Scripts\Activate.ps1
    ```
*   On **Windows** (Command Prompt):
    ```cmd
    .venv\Scripts\activate.bat
    ```

Once active, your terminal prompt will be prefixed with `(.venv)`. This tells you the environment is on.

**3. Install the exact dependencies** (only needed the first time, with the venv activated):
```bash
pip install -r requirements.txt
```
This downloads and installs the pinned versions listed in `requirements.txt`. It may take a few minutes.

> **Important:** Every time you open a *new* terminal to run Silkscan, you must re-activate the environment (step 2) first. If your prompt doesn't show `(.venv)`, the program will fall back to your system's Python and may crash. To leave the environment when you're done, type `deactivate`.

---

## 3. Launching the Interface

With your virtual environment **activated** (you should see `(.venv)` in your prompt), open the graphical interface by running:

```bash
python silkscan_gui.py
```

A window titled **"Silkscan 3D Reconstruction"** will pop up.

---

## 4. Processing Your Video

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

## 5. Cropping and Saving Your 3D Model

When processing finishes, the Tkinter window will close and an interactive **Silkscan Crop Editor** will automatically open.

1. **Adjust Boundaries:** Use the sliders on the left (X, Y, and Z Min/Max) to trim off any remaining pieces of the capture box walls. The 3D model will instantly update as you drag!
2. **Save Results:** Once your spider web is perfectly isolated, click **"Save All Thresholds"** (or "Save Cropped PCD" if using Steger). The editor will apply your exact bounding box crop and save your final 3D point cloud files to your selected output folder.

If you want to view the `.pcd` file later, you can use specialized 3D software like **MeshLab** or **CloudCompare** (both are free).
1. Download and install [MeshLab](https://www.meshlab.net/).
2. Open MeshLab.
3. Go to `File` -> `Import Mesh...` and select your `.pcd` file.

---

## 6. Troubleshooting

**"The program runs until the stacking is complete, then crashes (segmentation fault / SIGSEGV)!"**
This is almost always caused by running with **mismatched library versions** (a `numpy`/`open3d` ABI conflict) outside of the project's virtual environment.
**The Fix:** Make sure you followed **Section 2** and are running inside the activated `.venv` (your prompt shows `(.venv)`) with dependencies installed from `requirements.txt`. If you set the venv up before this fix was released, refresh it:
```bash
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```
If the crash persists, delete the `.venv` folder and recreate it from scratch using the steps in Section 2 to guarantee a clean, pinned install.

*   **Error: "ModuleNotFoundError: No module named 'tkinter'" (or 'No module named \_tkinter')**
    Tkinter is the GUI toolkit. It ships built-in with the [python.org](https://www.python.org/downloads/) installer, but **Homebrew's Python does not include it by default**. If you're on a Mac using Homebrew Python, install it with:
    ```bash
    brew install python-tk@3.11
    ```
    No need to recreate your venv afterward — just re-run `python silkscan_gui.py`. (If it still isn't found, delete and recreate the `.venv` as described in Section 2 so it re-links against the now-Tk-enabled Python.) The simplest way to avoid this entirely is to use the official python.org Python 3.11 installer.
*   **The point cloud looks stretched or flat.**
    This means your scaling parameters (`mm / frame` or `pixels / mm`) are incorrect. If you didn't have a `manifest.json`, you will need to manually adjust these values in the GUI based on your physical camera setup.
