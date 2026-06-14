import sys
import os
import json
import time
import multiprocessing
import numpy as np
import open3d as o3d
import tkinter as tk
from tkinter import ttk

def save_pcd(pcd_data, path):
    if len(pcd_data) == 0:
        return
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pcd_data[:, :3])
    if pcd_data.shape[1] >= 8:
        strengths = pcd_data[:, 7]
        denom = strengths.max() - strengths.min() + 1e-6
        norm_s = (strengths - strengths.min()) / denom
        colors = np.column_stack((norm_s, norm_s, norm_s))
        pcd.colors = o3d.utility.Vector3dVector(colors)
    o3d.io.write_point_cloud(path, pcd)

def viewer_process(pcd_path, bounds_file):
    import open3d as o3d
    import numpy as np
    import json
    import os
    import time
    
    # Fixed, high-contrast color for the points so they stay clearly visible
    # against the white preview background (strength-based grayscale would make
    # high-strength points blend into the background).
    PREVIEW_COLOR = [0.0, 0.2, 0.8]

    pcd_array = np.load(pcd_path)
    pcd = o3d.geometry.PointCloud()
    if len(pcd_array) > 0:
        pcd.points = o3d.utility.Vector3dVector(pcd_array[:, :3])
        pcd.paint_uniform_color(PREVIEW_COLOR)

    vis = o3d.visualization.Visualizer()
    vis.create_window("Silkscan 3D Preview (Legacy Mode)", width=1024, height=768)
    vis.add_geometry(pcd)
    
    # Optional: Set a nice background and point size
    opt = vis.get_render_option()
    opt.background_color = np.asarray([1.0, 1.0, 1.0])
    opt.point_size = 2.0
    
    last_bounds = None
    
    while True:
        vis.poll_events()
        vis.update_renderer()
        
        if os.path.exists(bounds_file):
            try:
                with open(bounds_file, 'r') as f:
                    bounds = json.load(f)
            except:
                bounds = last_bounds
                
            if bounds and bounds != last_bounds:
                mask = (pcd_array[:, 0] >= bounds['x_min']) & (pcd_array[:, 0] <= bounds['x_max']) & \
                       (pcd_array[:, 1] >= bounds['y_min']) & (pcd_array[:, 1] <= bounds['y_max']) & \
                       (pcd_array[:, 2] >= bounds['z_min']) & (pcd_array[:, 2] <= bounds['z_max'])
                filtered = pcd_array[mask]
                
                # Open3D legacy visualizer crashes if point cloud has 0 points
                if len(filtered) > 0:
                    pcd.points = o3d.utility.Vector3dVector(filtered[:, :3])
                    pcd.paint_uniform_color(PREVIEW_COLOR)
                    vis.update_geometry(pcd)
                last_bounds = bounds
                
        time.sleep(0.05)

class LegacyEditorApp:
    def __init__(self, root, pcd_path, method, output_path):
        self.root = root
        self.root.title("Silkscan Editor Controls")
        self.root.geometry("400x550")
        
        self.pcd_path = pcd_path
        self.method = method
        self.output_path = output_path
        self.bounds_file = output_path + ".bounds.json"
        
        self.pcd_array = np.load(pcd_path)
        
        if len(self.pcd_array) > 0:
            self.bounds = {
                'x_min': float(np.min(self.pcd_array[:, 0])),
                'x_max': float(np.max(self.pcd_array[:, 0])),
                'y_min': float(np.min(self.pcd_array[:, 1])),
                'y_max': float(np.max(self.pcd_array[:, 1])),
                'z_min': float(np.min(self.pcd_array[:, 2])),
                'z_max': float(np.max(self.pcd_array[:, 2]))
            }
        else:
            self.bounds = {'x_min': 0, 'x_max': 0, 'y_min': 0, 'y_max': 0, 'z_min': 0, 'z_max': 0}
            
        self.write_bounds()
        
        self.viewer_p = multiprocessing.Process(target=viewer_process, args=(self.pcd_path, self.bounds_file))
        self.viewer_p.start()
        
        self.create_widgets()
        
        # Ensure cleanup on close
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

    def write_bounds(self):
        try:
            with open(self.bounds_file, 'w') as f:
                json.dump(self.bounds, f)
        except Exception as e:
            print("Error writing bounds:", e)

    def create_widgets(self):
        main_frame = ttk.Frame(self.root, padding="20")
        main_frame.pack(fill=tk.BOTH, expand=True)

        ttk.Label(main_frame, text="Adjust Crop Boundaries", font=("Helvetica", 14, "bold")).pack(pady=(0, 20))

        def make_slider(name, key, min_val, max_val):
            frame = ttk.Frame(main_frame)
            frame.pack(fill=tk.X, pady=5)
            ttk.Label(frame, text=name, width=10).pack(side=tk.LEFT)
            
            # Avoid tk error if min == max
            if min_val >= max_val:
                min_val -= 1
                max_val += 1
                
            var = tk.DoubleVar(value=self.bounds[key])

            value_label = ttk.Label(frame, text=f"{self.bounds[key]:.2f}", width=8, anchor=tk.E)

            def on_change(v):
                self.bounds[key] = float(v)
                value_label.config(text=f"{float(v):.2f}")
                self.write_bounds()

            slider = ttk.Scale(frame, from_=min_val, to=max_val, variable=var, orient=tk.HORIZONTAL, command=on_change)
            slider.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
            value_label.pack(side=tk.LEFT)

        make_slider("X Min", 'x_min', self.bounds['x_min'], self.bounds['x_max'])
        make_slider("X Max", 'x_max', self.bounds['x_min'], self.bounds['x_max'])
        make_slider("Y Min", 'y_min', self.bounds['y_min'], self.bounds['y_max'])
        make_slider("Y Max", 'y_max', self.bounds['y_min'], self.bounds['y_max'])
        make_slider("Z Min", 'z_min', self.bounds['z_min'], self.bounds['z_max'])
        make_slider("Z Max", 'z_max', self.bounds['z_min'], self.bounds['z_max'])

        btn_text = "Save All Thresholds" if self.method == "Threshold" else "Save Cropped PCD"
        save_btn = ttk.Button(main_frame, text=btn_text, command=self.on_save)
        save_btn.pack(pady=30, fill=tk.X)
        
        self.status_var = tk.StringVar(value="")
        ttk.Label(main_frame, textvariable=self.status_var).pack(pady=5)

    def on_save(self):
        self.status_var.set("Applying final crop...")
        self.root.update()
        
        mask = (self.pcd_array[:, 0] >= self.bounds['x_min']) & (self.pcd_array[:, 0] <= self.bounds['x_max']) & \
               (self.pcd_array[:, 1] >= self.bounds['y_min']) & (self.pcd_array[:, 1] <= self.bounds['y_max']) & \
               (self.pcd_array[:, 2] >= self.bounds['z_min']) & (self.pcd_array[:, 2] <= self.bounds['z_max'])
        cropped = self.pcd_array[mask]
        
        if self.method == "Threshold":
            base, ext = os.path.splitext(self.output_path)
            thresholds = np.arange(0.15, 0.95, 0.05)
            for t in thresholds:
                t_data = cropped[cropped[:, 3] >= t]
                t_output = f"{base}_{t:.2f}{ext}"
                save_pcd(t_data, t_output)
                print(f"Saved threshold {t:.2f} ({len(t_data)} pts) to {os.path.basename(t_output)}")
            print("Successfully saved all thresholds!")
        else:
            save_pcd(cropped, self.output_path)
            print(f"Saved ({len(cropped)} pts) to {self.output_path}")
            
        self.on_close()

    def on_close(self):
        if self.viewer_p and self.viewer_p.is_alive():
            self.viewer_p.terminate()
        if os.path.exists(self.bounds_file):
            os.remove(self.bounds_file)
        self.root.destroy()

if __name__ == "__main__":
    if len(sys.argv) < 4:
        print("Usage: legacy_editor_app.py <pcd_npy_path> <method> <output_path>")
        sys.exit(1)
        
    pcd_path = sys.argv[1]
    method = sys.argv[2]
    output_path = sys.argv[3]
    
    root = tk.Tk()
    
    # Use clam theme
    style = ttk.Style()
    style.theme_use('clam')
    
    app = LegacyEditorApp(root, pcd_path, method, output_path)
    root.mainloop()
