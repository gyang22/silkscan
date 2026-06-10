import sys
import os
import numpy as np

# Open3D imports
import open3d as o3d
import open3d.visualization.gui as gui
import open3d.visualization.rendering as rendering

def save_pcd(pcd_data, path):
    if len(pcd_data) == 0:
        print("Empty point cloud, not saving.")
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

def run_o3d_editor(pcd_array, method, output_path):
    app = gui.Application.instance
    app.initialize()

    window = app.create_window("Silkscan Crop Editor", 1200, 800)

    widget3d = gui.SceneWidget()
    widget3d.scene = rendering.Open3DScene(window.renderer)

    pcd = o3d.geometry.PointCloud()
    if len(pcd_array) > 0:
        pcd.points = o3d.utility.Vector3dVector(pcd_array[:, :3])
    
    mat = rendering.MaterialRecord()
    mat.shader = "defaultUnlit"
    mat.point_size = 2.0
    widget3d.scene.add_geometry("PCD", pcd, mat)
    
    if len(pcd_array) > 0:
        bbox = widget3d.scene.bounding_box
        widget3d.setup_camera(60.0, bbox, bbox.get_center())

    panel = gui.Vert(10, gui.Margins(10, 10, 10, 10))
    panel.add_child(gui.Label("Adjust Crop Boundaries"))

    if len(pcd_array) > 0:
        bounds = {
            'x_min': float(np.min(pcd_array[:, 0])),
            'x_max': float(np.max(pcd_array[:, 0])),
            'y_min': float(np.min(pcd_array[:, 1])),
            'y_max': float(np.max(pcd_array[:, 1])),
            'z_min': float(np.min(pcd_array[:, 2])),
            'z_max': float(np.max(pcd_array[:, 2]))
        }
    else:
        bounds = {'x_min': 0, 'x_max': 0, 'y_min': 0, 'y_max': 0, 'z_min': 0, 'z_max': 0}

    current_bounds = bounds.copy()

    def update_geometry():
        mask = (pcd_array[:, 0] >= current_bounds['x_min']) & (pcd_array[:, 0] <= current_bounds['x_max']) & \
               (pcd_array[:, 1] >= current_bounds['y_min']) & (pcd_array[:, 1] <= current_bounds['y_max']) & \
               (pcd_array[:, 2] >= current_bounds['z_min']) & (pcd_array[:, 2] <= current_bounds['z_max'])
        filtered = pcd_array[mask]
        
        new_pcd = o3d.geometry.PointCloud()
        if len(filtered) > 0:
            new_pcd.points = o3d.utility.Vector3dVector(filtered[:, :3])
        
        widget3d.scene.remove_geometry("PCD")
        widget3d.scene.add_geometry("PCD", new_pcd, mat)

    def make_slider(name, key, min_val, max_val, is_min):
        panel.add_child(gui.Label(name))
        slider = gui.Slider(gui.Slider.DOUBLE)
        if min_val >= max_val:
            slider.set_limits(min_val - 1.0, min_val + 1.0)
        else:
            slider.set_limits(min_val, max_val)
        slider.double_value = current_bounds[key]
        def on_change(val):
            current_bounds[key] = val
            update_geometry()
        slider.set_on_value_changed(on_change)
        panel.add_child(slider)

    if len(pcd_array) > 0:
        make_slider("X Min", 'x_min', bounds['x_min'], bounds['x_max'], True)
        make_slider("X Max", 'x_max', bounds['x_min'], bounds['x_max'], False)
        make_slider("Y Min", 'y_min', bounds['y_min'], bounds['y_max'], True)
        make_slider("Y Max", 'y_max', bounds['y_min'], bounds['y_max'], False)
        make_slider("Z Min", 'z_min', bounds['z_min'], bounds['z_max'], True)
        make_slider("Z Max", 'z_max', bounds['z_min'], bounds['z_max'], False)

    btn_text = "Save All Thresholds" if method == "Threshold" else "Save Cropped PCD"
    save_btn = gui.Button(btn_text)
    
    def on_save():
        print("Applying final crop...")
        mask = (pcd_array[:, 0] >= current_bounds['x_min']) & (pcd_array[:, 0] <= current_bounds['x_max']) & \
               (pcd_array[:, 1] >= current_bounds['y_min']) & (pcd_array[:, 1] <= current_bounds['y_max']) & \
               (pcd_array[:, 2] >= current_bounds['z_min']) & (pcd_array[:, 2] <= current_bounds['z_max'])
        cropped = pcd_array[mask]
        
        if method == "Threshold":
            base, ext = os.path.splitext(output_path)
            thresholds = np.arange(0.15, 0.95, 0.05)
            for t in thresholds:
                t_data = cropped[cropped[:, 3] >= t]
                t_output = f"{base}_{t:.2f}{ext}"
                save_pcd(t_data, t_output)
                print(f"Saved threshold {t:.2f} ({len(t_data)} pts) to {os.path.basename(t_output)}")
            print("Successfully saved all thresholds!")
        else:
            save_pcd(cropped, output_path)
            print(f"Saved ({len(cropped)} pts) to {output_path}")
            
        app.quit()
        
    save_btn.set_on_clicked(on_save)
    panel.add_child(save_btn)

    window.add_child(panel)
    window.add_child(widget3d)

    def on_layout(layout_context):
        r = window.content_rect
        panel_width = 300
        panel.frame = gui.Rect(r.x, r.y, panel_width, r.height)
        widget3d.frame = gui.Rect(r.x + panel_width, r.y, r.width - panel_width, r.height)

    window.set_on_layout(on_layout)
    app.run()

if __name__ == "__main__":
    if len(sys.argv) < 4:
        print("Usage: o3d_editor_app.py <pcd_npy_path> <method> <output_path>")
        sys.exit(1)
        
    pcd_path = sys.argv[1]
    method = sys.argv[2]
    output_path = sys.argv[3]
    
    pcd_array = np.load(pcd_path)
    run_o3d_editor(pcd_array, method, output_path)
