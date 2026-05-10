import json
import os

class CaptureSet:
    def __init__(self, directory):
        self.directory = directory
        manifest_path = os.path.join(directory, 'manifest.json')
        with open(manifest_path, 'r') as f:
            self.manifest = json.load(f)
            
        self.name = self.manifest.get('name', 'unknown')
        self.pixels_per_mm = self.manifest.get('pixels_per_mm', 1.0)
        self.mm_per_frame = self.manifest.get('mm_per_frame', 0.1)
        self.sweeps = self.manifest.get('sweeps', [])
        
class Config:
    def __init__(self,
                 method='steger',
                 intensity_threshold=0.05,
                 strength_threshold=0.002,
                 high_intensity_threshold=0.15,
                 high_strength_threshold=0.015,
                 sigma=0.5,
                 persistence_min_frames=2,
                 temporal_spatial_radius=5.0,
                 temporal_max_gap_frames=2,
                 dedup_radius_mm=0.5,
                 icp_voxel_size=2.0,
                 icp_distance_threshold=5.0,
                 quality_score_min=0.3,
                 box_crop_padding_px=25,
                 spatial_2d_min_length_px=20,
                 temporal_stack_frames=0):
        self.method = method
        self.intensity_threshold = intensity_threshold
        self.strength_threshold = strength_threshold
        self.high_intensity_threshold = high_intensity_threshold
        self.high_strength_threshold = high_strength_threshold
        self.sigma = sigma
        self.persistence_min_frames = persistence_min_frames
        self.temporal_spatial_radius = temporal_spatial_radius
        self.temporal_max_gap_frames = temporal_max_gap_frames
        self.dedup_radius_mm = dedup_radius_mm
        self.icp_voxel_size = icp_voxel_size
        self.icp_distance_threshold = icp_distance_threshold
        self.quality_score_min = quality_score_min
        self.box_crop_padding_px = box_crop_padding_px
        self.spatial_2d_min_length_px = spatial_2d_min_length_px
        self.temporal_stack_frames = temporal_stack_frames

def load_capture_set(directory):
    return CaptureSet(directory)
