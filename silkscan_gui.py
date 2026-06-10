#!/usr/bin/env python3
import tkinter as tk
from tkinter import filedialog, messagebox, ttk, scrolledtext
import os
import sys
import threading
import json

class ConsoleRedirector:
    def __init__(self, text_widget):
        self.text_widget = text_widget

    def write(self, str_data):
        self.text_widget.after(0, self._write, str_data)

    def _write(self, str_data):
        self.text_widget.configure(state='normal')
        parts = str_data.split('\r')
        if len(parts) > 1:
            self.text_widget.delete("end-1c linestart", "end-1c")
            self.text_widget.insert("end", parts[-1])
        else:
            self.text_widget.insert("end", str_data)
        self.text_widget.see("end")
        self.text_widget.configure(state='disabled')

    def flush(self):
        pass

# Add the current directory to sys.path so silkscan can be imported
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

import silkscan
from silkscan import SweepProcessor, get_sum_image_cached, detect_quad_rotated

class SilkscanGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Silkscan 3D Reconstruction")
        self.root.geometry("650x700")
        
        # Variables
        self.video_path_var = tk.StringVar()
        self.output_path_var = tk.StringVar(value=os.path.abspath("output.pcd"))
        self.mm_per_frame_var = tk.DoubleVar(value=0.067285)
        self.pixels_per_mm_var = tk.DoubleVar(value=3.4)
        self.start_frame_var = tk.IntVar(value=0)
        self.max_frames_var = tk.StringVar(value="") # Empty means all
        self.auto_crop_var = tk.BooleanVar(value=True)
        self.crop_padding_var = tk.IntVar(value=50)
        self.method_var = tk.StringVar(value="Threshold")
        self.launch_editor_data = None
        
        self.create_widgets()

    def create_widgets(self):
        main_frame = ttk.Frame(self.root, padding="20")
        main_frame.pack(fill=tk.BOTH, expand=True)

        # Title
        title_label = ttk.Label(main_frame, text="Silkscan Processor", font=("Helvetica", 16, "bold"))
        title_label.grid(row=0, column=0, columnspan=3, pady=(0, 20))

        # --- File Selection ---
        ttk.Label(main_frame, text="Video File:").grid(row=1, column=0, sticky=tk.W, pady=5)
        ttk.Entry(main_frame, textvariable=self.video_path_var, width=40).grid(row=1, column=1, padx=5)
        ttk.Button(main_frame, text="Browse", command=self.browse_video).grid(row=1, column=2)

        ttk.Label(main_frame, text="Output PCD:").grid(row=2, column=0, sticky=tk.W, pady=5)
        ttk.Entry(main_frame, textvariable=self.output_path_var, width=40).grid(row=2, column=1, padx=5)
        ttk.Button(main_frame, text="Browse", command=self.browse_output).grid(row=2, column=2)

        # --- Parameters ---
        param_frame = ttk.LabelFrame(main_frame, text="Scaling Parameters", padding="10")
        param_frame.grid(row=3, column=0, columnspan=3, sticky=tk.EW, pady=15)

        ttk.Label(param_frame, text="mm / frame:").grid(row=0, column=0, sticky=tk.W, pady=5)
        ttk.Entry(param_frame, textvariable=self.mm_per_frame_var, width=15).grid(row=0, column=1, padx=5, sticky=tk.W)

        ttk.Label(param_frame, text="pixels / mm:").grid(row=0, column=2, sticky=tk.W, pady=5, padx=(20,0))
        ttk.Entry(param_frame, textvariable=self.pixels_per_mm_var, width=15).grid(row=0, column=3, padx=5, sticky=tk.W)

        # --- Frames ---
        frame_param_frame = ttk.LabelFrame(main_frame, text="Frame Range (Optional)", padding="10")
        frame_param_frame.grid(row=4, column=0, columnspan=3, sticky=tk.EW, pady=5)

        ttk.Label(frame_param_frame, text="Start Frame:").grid(row=0, column=0, sticky=tk.W, pady=5)
        ttk.Entry(frame_param_frame, textvariable=self.start_frame_var, width=15).grid(row=0, column=1, padx=5, sticky=tk.W)

        ttk.Label(frame_param_frame, text="Max Frames:").grid(row=0, column=2, sticky=tk.W, pady=5, padx=(20,0))
        ttk.Entry(frame_param_frame, textvariable=self.max_frames_var, width=15).grid(row=0, column=3, padx=5, sticky=tk.W)
        ttk.Label(frame_param_frame, text="(leave blank for all)").grid(row=0, column=4, sticky=tk.W, padx=5)

        # --- Options ---
        options_frame = ttk.LabelFrame(main_frame, text="Options", padding="10")
        options_frame.grid(row=5, column=0, columnspan=3, sticky=tk.EW, pady=15)
        
        ttk.Checkbutton(options_frame, text="Enable Auto-Cropping (Mask out background)", variable=self.auto_crop_var).grid(row=0, column=0, sticky=tk.W, pady=2)
        
        crop_pad_frame = ttk.Frame(options_frame)
        crop_pad_frame.grid(row=1, column=0, sticky=tk.W, pady=2, padx=20)
        ttk.Label(crop_pad_frame, text="Auto-Crop Padding (px):").pack(side=tk.LEFT)
        ttk.Entry(crop_pad_frame, textvariable=self.crop_padding_var, width=10).pack(side=tk.LEFT, padx=5)

        method_frame = ttk.Frame(options_frame)
        method_frame.grid(row=2, column=0, sticky=tk.W, pady=2)
        ttk.Label(method_frame, text="Extraction Method:").pack(side=tk.LEFT)
        method_cb = ttk.Combobox(method_frame, textvariable=self.method_var, values=["Steger", "Threshold"], state="readonly", width=15)
        method_cb.pack(side=tk.LEFT, padx=5)

        # --- Run Button & Status ---
        btn_frame = ttk.Frame(main_frame)
        btn_frame.grid(row=6, column=0, columnspan=3, pady=10)
        
        self.run_btn = ttk.Button(btn_frame, text="Start Processing & Open Editor", command=self.start_processing, style="Accent.TButton")
        self.run_btn.pack(side=tk.LEFT, padx=5)

        self.status_var = tk.StringVar(value="Ready")
        self.status_label = ttk.Label(main_frame, textvariable=self.status_var, font=("Helvetica", 10, "italic"))
        self.status_label.grid(row=7, column=0, columnspan=3, pady=5)
        
        self.console_text = scrolledtext.ScrolledText(main_frame, height=12, width=75, state='disabled', bg='black', fg='white', font=("Courier", 10))
        self.console_text.grid(row=8, column=0, columnspan=3, pady=10, sticky=tk.EW)

    def browse_video(self):
        filepath = filedialog.askopenfilename(filetypes=[("Video files", "*.mp4 *.avi")])
        if filepath:
            self.video_path_var.set(filepath)
            self.try_load_manifest(filepath)
            
            # Suggest output path
            base, _ = os.path.splitext(filepath)
            self.output_path_var.set(base + ".pcd")

    def browse_output(self):
        filepath = filedialog.asksaveasfilename(defaultextension=".pcd", filetypes=[("Point Cloud", "*.pcd")])
        if filepath:
            self.output_path_var.set(filepath)

    def try_load_manifest(self, video_path):
        directory = os.path.dirname(os.path.abspath(video_path))
        manifest_path = os.path.join(directory, 'manifest.json')
        if os.path.exists(manifest_path):
            try:
                with open(manifest_path, 'r') as f:
                    manifest = json.load(f)
                if 'mm_per_frame' in manifest:
                    self.mm_per_frame_var.set(manifest['mm_per_frame'])
                if 'pixels_per_mm' in manifest:
                    self.pixels_per_mm_var.set(manifest['pixels_per_mm'])
                self.status_var.set("Successfully loaded scaling parameters from manifest.json")
            except Exception as e:
                self.status_var.set(f"Could not load manifest.json: {e}")
        else:
            self.status_var.set("No manifest.json found. Using default scaling.")

    def start_processing(self):
        video_path = self.video_path_var.get()
        if not video_path or not os.path.exists(video_path):
            messagebox.showerror("Error", "Please select a valid video file.")
            return

        self.run_btn.config(state=tk.DISABLED)
        self.status_var.set("Processing... Please wait (this may take several minutes).")
        
        self.console_text.configure(state='normal')
        self.console_text.delete(1.0, tk.END)
        self.console_text.configure(state='disabled')
        
        self.old_stdout = sys.stdout
        self.old_stderr = sys.stderr
        redirector = ConsoleRedirector(self.console_text)
        sys.stdout = redirector
        sys.stderr = redirector
        
        # Run in background thread to keep GUI responsive
        thread = threading.Thread(target=self.process_video_thread, args=(video_path,))
        thread.daemon = True
        thread.start()

    def process_video_thread(self, video_path):
        try:
            manifest = {
                'pixels_per_mm': self.pixels_per_mm_var.get(),
                'mm_per_frame': self.mm_per_frame_var.get(),
                'crop_box_mm': {}
            }
            
            sweep_info = {
                'id': 'gui_sweep',
                'video_path': video_path,
                'rotation_angle_deg': 0.0
            }

            max_frames_str = self.max_frames_var.get().strip()
            max_frames = int(max_frames_str) if max_frames_str else None
            
            override_mask_poly = None
            if self.auto_crop_var.get():
                self.status_var.set("Detecting Auto-Crop Mask from background...")
                directory = os.path.dirname(os.path.abspath(video_path))
                cache_dir = os.path.join(directory, '.cache')
                os.makedirs(cache_dir, exist_ok=True)
                
                import cv2
                cap = cv2.VideoCapture(video_path)
                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                cap.release()
                end_frame = (self.start_frame_var.get() + max_frames - 1) if max_frames else (total_frames - 1)
                
                accum = get_sum_image_cached(sweep_info['id'], video_path, self.start_frame_var.get(), end_frame, cache_dir, step=30)
                if accum is not None:
                    res = detect_quad_rotated(accum, pad=self.crop_padding_var.get())
                    override_mask_poly = res['quad']
                    print(f"Auto-Crop found best angle: {res['best_angle']:.1f} degrees")
                    print(f"Auto-Crop bounds (xl, xr, yt, yb): {res['xl']:.1f}, {res['xr']:.1f}, {res['yt']:.1f}, {res['yb']:.1f}")
                else:
                    self.status_var.set("Warning: Auto-Crop failed. Processing full image.")

            method = self.method_var.get()
            output_path = self.output_path_var.get()
            os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
            
            if method == "Steger":
                self.status_var.set("Extracting 3D Point Cloud (Steger). Please wait...")
                config = silkscan.Config(
                    method='steger', 
                    intensity_threshold=0.05, strength_threshold=0.002,
                    high_intensity_threshold=0.15, high_strength_threshold=0.015,
                    sigma=0.5, persistence_min_frames=5, temporal_spatial_radius=2.0,
                    spatial_2d_min_length_px=20, temporal_stack_frames=5,
                    box_crop_padding_px=self.crop_padding_var.get()
                )
                processor = SweepProcessor(config)
                pcd_data = processor.process_video(
                    video_path, sweep_info, manifest, 
                    start_frame=self.start_frame_var.get(), max_frames=max_frames, override_mask_poly=override_mask_poly
                )
                self.launch_editor_data = (pcd_data, method, output_path)
            
            elif method == "Threshold":
                import numpy as np
                self.status_var.set(f"Extracting base Point Cloud (Threshold 0.15). Please wait...")
                config = silkscan.Config(
                    method='threshold', 
                    intensity_threshold=0.15,
                    persistence_min_frames=5, temporal_spatial_radius=2.0,
                    spatial_2d_min_length_px=20, temporal_stack_frames=5,
                    box_crop_padding_px=self.crop_padding_var.get()
                )
                processor = SweepProcessor(config)
                base_pcd = processor.process_video(
                    video_path, sweep_info, manifest, 
                    start_frame=self.start_frame_var.get(), max_frames=max_frames, override_mask_poly=override_mask_poly
                )
                self.launch_editor_data = (base_pcd, method, output_path)
            
            self.root.after(0, self.on_processing_complete_and_launch)
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            self.root.after(0, messagebox.showerror, "Processing Error", str(e))
            self.status_var.set("An error occurred.")
        finally:
            sys.stdout = self.old_stdout
            sys.stderr = self.old_stderr
            self.root.after(0, lambda: self.run_btn.config(state=tk.NORMAL))

    def on_processing_complete_and_launch(self):
        self.status_var.set("Finished processing. Launching 3D Editor...")
        self.root.destroy()

def run_o3d_editor(pcd_array, method, output_path):
    import subprocess
    import numpy as np
    
    editor_script = os.path.join(os.path.dirname(__file__), 'silkscan', 'o3d_editor_app.py')
    
    try:
        import open3d.visualization.gui as gui
        python_exe = sys.executable
    except ModuleNotFoundError:
        print("Warning: open3d.visualization.gui not found in your conda environment.")
        print("Falling back to system python3 to launch the 3D editor...")
        python_exe = "/usr/local/bin/python3"
        
    temp_data = output_path + ".temp.npy"
    np.save(temp_data, pcd_array)
    
    try:
        subprocess.run([python_exe, editor_script, temp_data, method, output_path])
    finally:
        if os.path.exists(temp_data):
            os.remove(temp_data)

if __name__ == "__main__":
    root = tk.Tk()
    
    style = ttk.Style()
    style.theme_use('clam')
    
    app = SilkscanGUI(root)
    root.mainloop()

    if app.launch_editor_data is not None:
        pcd_data, method, output_path = app.launch_editor_data
        run_o3d_editor(pcd_data, method, output_path)
