import json
import cv2
import numpy as np
import os
from skimage.feature import hessian_matrix, hessian_matrix_eigvals
from scipy.spatial import cKDTree, ConvexHull, Delaunay
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from scipy.ndimage import gaussian_filter1d
import open3d as o3d
from tqdm import tqdm

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

def simple_threshold_detection(image, intensity_threshold=0.15):
    mask = (image >= intensity_threshold)
    r_idx, c_idx = np.where(mask)
    if len(r_idx) == 0:
        return np.zeros((0, 7))
        
    intensities = image[r_idx, c_idx]
    strengths = intensities
    n_c = np.zeros_like(c_idx, dtype=np.float32)
    n_r = np.zeros_like(r_idx, dtype=np.float32)
    is_high_conf = np.ones_like(c_idx, dtype=bool)
    
    return np.column_stack((
        c_idx, 
        r_idx, 
        intensities, 
        strengths, 
        n_c, 
        n_r,
        is_high_conf
    ))

def stegers_line_detection(image, 
                           sigma=0.5, 
                           intensity_threshold=0.05, 
                           strength_threshold=0.002, 
                           high_intensity_threshold=0.15, 
                           high_strength_threshold=0.015,
                           spatial_2d_min_length_px=20):
    """
    Hybrid subpixel line detection using Steger's method.
    
    Strategy: bright pixels (>= high_intensity_threshold) are kept unconditionally,
    just like simple thresholding. Faint pixels (between intensity_threshold and
    high_intensity_threshold) are kept only if Steger's ridge NMS confirms they
    lie on a true ridge centerline. This makes the output a strict superset of
    simple thresholding while extending into the faint sub-threshold region.
    
    Returns array of (x, y, intensity, line_strength, n_x, n_y, is_high_conf)
    """
    # Use low intensity gate to find ALL candidate pixels (bright + faint)
    mask = (image >= intensity_threshold)
    
    r_idx, c_idx = np.where(mask)
    if len(r_idx) == 0:
        return np.zeros((0, 7))
        
    intensities = image[r_idx, c_idx]
    is_bright = intensities >= high_intensity_threshold
        
    # 1. Smooth and compute Hessian
    H_elems = hessian_matrix(image, sigma=sigma, order='rc')
    
    # 2. Compute eigenvalues and eigenvectors
    # eigvals returned in DECREASING order; eigvals[1] is most negative.
    # For bright ridges, the curvature across the ridge is large & negative.
    eigvals = hessian_matrix_eigvals(H_elems)
    lambda_min = eigvals[1]
    
    H_rr, H_rc, H_cc = H_elems
    
    # Eigenvector for lambda_min: perpendicular to the ridge
    v_r = -H_rc[r_idx, c_idx]
    v_c = H_rr[r_idx, c_idx] - lambda_min[r_idx, c_idx]
    
    norm = np.sqrt(v_r**2 + v_c**2)
    norm[norm == 0] = 1.0
    n_r = v_r / norm
    n_c = v_c / norm
    
    # Compute first derivatives for subpixel offset
    grad_r = gaussian_filter1d(image, sigma=sigma, axis=0, order=1)
    grad_r = gaussian_filter1d(grad_r, sigma=sigma, axis=1, order=0)
    
    grad_c = gaussian_filter1d(image, sigma=sigma, axis=1, order=1)
    grad_c = gaussian_filter1d(grad_c, sigma=sigma, axis=0, order=0)
    
    g_r = grad_r[r_idx, c_idx]
    g_c = grad_c[r_idx, c_idx]
    
    g_dot_n = g_r * n_r + g_c * n_c
    
    denom = lambda_min[r_idx, c_idx]
    
    t = np.zeros_like(denom)
    valid_denom = np.abs(denom) > 1e-6
    t[valid_denom] = - g_dot_n[valid_denom] / denom[valid_denom]
    
    # Steger NMS: ridge centerline passes through pixel AND negative curvature
    is_ridge = (np.abs(t) <= 0.5) & (denom < 0)
    
    # HYBRID SELECTION:
    # - Bright pixels: keep ALL (like simple thresholding) — clamp t for subpixel position
    # - Faint pixels: keep ONLY if they pass the Steger ridge NMS test
    keep = is_bright | (is_ridge & ~is_bright)
    
    r_idx = r_idx[keep]
    c_idx = c_idx[keep]
    n_r = n_r[keep]
    n_c = n_c[keep]
    
    # For bright pixels that didn't pass NMS, clamp t so they still get a position
    t_keep = t[keep]
    bright_kept = is_bright[keep]
    t_keep[bright_kept] = np.clip(t_keep[bright_kept], -0.5, 0.5)
    
    if len(r_idx) == 0:
        return np.zeros((0, 7))
    
    r_sub = r_idx + t_keep * n_r
    c_sub = c_idx + t_keep * n_c
    
    intensities = image[r_idx, c_idx]
    strengths = np.abs(lambda_min[r_idx, c_idx])
    
    # For faint points, also require minimum line strength
    is_faint = intensities < high_intensity_threshold
    valid = ~is_faint | (strengths >= strength_threshold)
    
    pts = np.column_stack((c_sub[valid], r_sub[valid]))
    valid_intensities = intensities[valid]
    valid_strengths = strengths[valid]
    n_c_valid = n_c[valid]
    n_r_valid = n_r[valid]
    
    # 2D spatial length filter: remove isolated short noise ridges
    if spatial_2d_min_length_px > 1 and len(pts) > 0:
        # Only apply length filter to faint points — never remove bright ones
        faint_mask = valid_intensities < high_intensity_threshold
        
        if np.any(faint_mask):
            faint_pts = pts[faint_mask]
            tree = cKDTree(faint_pts)
            pairs = tree.query_pairs(r=1.5)
            
            if pairs:
                pairs_arr = np.array(list(pairs))
                V = np.ones(len(pairs_arr), dtype=bool)
                adj = csr_matrix((V, (pairs_arr[:, 0], pairs_arr[:, 1])), shape=(len(faint_pts), len(faint_pts)))
                _, labels = connected_components(csgraph=adj, directed=False)
                
                unique_labels, counts = np.unique(labels, return_counts=True)
                valid_labels = unique_labels[counts >= spatial_2d_min_length_px]
                faint_keep = np.isin(labels, valid_labels)
            else:
                faint_keep = np.zeros(np.sum(faint_mask), dtype=bool)
            
            # Combine: all bright points + filtered faint points
            final_keep = np.ones(len(pts), dtype=bool)
            final_keep[faint_mask] = faint_keep
            
            pts = pts[final_keep]
            valid_intensities = valid_intensities[final_keep]
            valid_strengths = valid_strengths[final_keep]
            n_c_valid = n_c_valid[final_keep]
            n_r_valid = n_r_valid[final_keep]
        
    if len(pts) == 0:
        return np.zeros((0, 7))
        
    is_high_conf = (valid_intensities >= high_intensity_threshold) & (valid_strengths >= high_strength_threshold)
    
    return np.column_stack((
        pts[:, 0], 
        pts[:, 1], 
        valid_intensities, 
        valid_strengths, 
        n_c_valid, 
        n_r_valid,
        is_high_conf.astype(np.float32)
    ))

import itertools

def temporal_coherence_filter(frames_detections, min_frames=2, spatial_radius=2.0, max_gap_frames=1):
    if not frames_detections:
        return []
        
    num_frames = len(frames_detections)
    trees = []
    kept = []
    
    # 1. Initialize
    for dets in frames_detections:
        if len(dets) > 0:
            trees.append(cKDTree(dets[:, :2]))
            kept.append(dets[:, 6] > 0.5) # is_high_conf
        else:
            trees.append(None)
            kept.append(np.array([], dtype=bool))
            
    # 2. Forward Pass (Vectorized Hysteresis Propagation)
    print("  Hysteresis Forward Pass...")
    for f in tqdm(range(num_frames), desc="Forward Propagation"):
        if not np.any(kept[f]):
            continue
            
        active_pts = frames_detections[f][kept[f], :2]
        
        for k in range(1, max_gap_frames + 1):
            n_f = f + k
            if n_f >= num_frames or trees[n_f] is None:
                continue
                
            idxs = trees[n_f].query_ball_point(active_pts, r=spatial_radius * k)
            
            if len(idxs) > 0:
                flat_idxs = np.fromiter(itertools.chain.from_iterable(idxs), dtype=int)
                if len(flat_idxs) > 0:
                    kept[n_f][flat_idxs] = True
                    
    # 3. Backward Pass
    print("  Hysteresis Backward Pass...")
    for f in tqdm(range(num_frames - 1, -1, -1), desc="Backward Propagation"):
        if not np.any(kept[f]):
            continue
            
        active_pts = frames_detections[f][kept[f], :2]
        
        for k in range(1, max_gap_frames + 1):
            n_f = f - k
            if n_f < 0 or trees[n_f] is None:
                continue
                
            idxs = trees[n_f].query_ball_point(active_pts, r=spatial_radius * k)
            
            if len(idxs) > 0:
                flat_idxs = np.fromiter(itertools.chain.from_iterable(idxs), dtype=int)
                if len(flat_idxs) > 0:
                    kept[n_f][flat_idxs] = True
                    
    # 4. Build Filtered Subgraph for Persistence Checking
    print("  Extracting components for persistence check...")
    surviving_trees = []
    global_offsets = np.zeros(num_frames, dtype=int)
    node_frames = []
    node_orig_idx = []
    
    current_offset = 0
    for f in range(num_frames):
        global_offsets[f] = current_offset
        if not np.any(kept[f]):
            surviving_trees.append(None)
            continue
            
        surviving_idx = np.where(kept[f])[0]
        surviving_pts = frames_detections[f][surviving_idx, :2]
        surviving_trees.append(cKDTree(surviving_pts))
        
        node_frames.append(np.full(len(surviving_idx), f))
        node_orig_idx.append(surviving_idx)
        current_offset += len(surviving_idx)
        
    num_nodes = current_offset
    
    if num_nodes == 0:
        return [np.zeros((0, 7)) for _ in range(num_frames)]
        
    node_frames = np.concatenate(node_frames)
    node_orig_idx = np.concatenate(node_orig_idx)
    
    I = []
    J = []
    
    # Vectorized Graph Edges on Surviving Points
    for f in tqdm(range(num_frames - 1), desc="Building Graph"):
        if surviving_trees[f] is None:
            continue
            
        surviving_idx_f = np.where(kept[f])[0]
        surviving_pts_f = frames_detections[f][surviving_idx_f, :2]
        
        for k in range(1, max_gap_frames + 1):
            n_f = f + k
            if n_f >= num_frames or surviving_trees[n_f] is None:
                continue
                
            surviving_idx_nf = np.where(kept[n_f])[0]
            n_pts_nf = len(surviving_idx_nf)
            
            # Use bounded k-nearest neighbor query to prevent O(N^2) dense blob edge explosions
            dists, idxs = surviving_trees[n_f].query(surviving_pts_f, k=3, distance_upper_bound=spatial_radius * k)
            
            if idxs.ndim == 1:
                idxs = idxs.reshape(-1, 1)
                
            valid_mask = idxs < n_pts_nf
            if not np.any(valid_mask):
                continue
                
            u_indices = np.repeat(np.arange(len(surviving_pts_f)), idxs.shape[1]).reshape(idxs.shape)
            
            u_valid = u_indices[valid_mask]
            v_valid = idxs[valid_mask]
            
            u_global = global_offsets[f] + u_valid
            v_global = global_offsets[n_f] + v_valid
            
            I.append(u_global)
            J.append(v_global)
            
    if I:
        I_arr = np.concatenate(I)
        J_arr = np.concatenate(J)
        V_arr = np.ones(len(I_arr), dtype=bool)
        
        adj = csr_matrix((V_arr, (I_arr, J_arr)), shape=(num_nodes, num_nodes))
        
        print("  Computing Connected Components...")
        n_components, labels = connected_components(csgraph=adj, directed=False, return_labels=True)
    else:
        labels = np.arange(num_nodes)
        n_components = num_nodes
                    
    # 5. Component Persistence Filter
    print("  Filtering by Persistence...")
    pairs = np.column_stack((labels, node_frames))
    unique_pairs = np.unique(pairs, axis=0)
    unique_labels, counts = np.unique(unique_pairs[:, 0], return_counts=True)
    
    pers_map = np.zeros(n_components, dtype=int)
    pers_map[unique_labels] = counts
    
    node_pers = pers_map[labels]
    keep_node_mask = node_pers >= min_frames
    
    filtered_detections = []
    for f in range(num_frames):
        mask_f = (node_frames == f) & keep_node_mask
        
        if not np.any(mask_f):
            filtered_detections.append(np.zeros((0, 7)))
            continue
            
        idx_to_keep = node_orig_idx[mask_f]
        pers_arr = node_pers[mask_f]
        
        dets_with_pers = np.column_stack((frames_detections[f][idx_to_keep, :6], pers_arr))
        filtered_detections.append(dets_with_pers)
        
    return filtered_detections

class SweepProcessor:
    """Processes a single video sweep into a 3D Point Cloud."""
    def __init__(self, config):
        self.config = config
        
    def process_video(self, video_path, sweep_info, manifest, start_frame=0, max_frames=None,
                       override_mask_poly=None):
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Could not open video {video_path}")
            
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        if start_frame > 0:
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
            
        process_count = total_frames - start_frame
        if max_frames is not None:
            process_count = min(process_count, max_frames)
        
        # Build 2D projected mask from 3D world crop box
        crop = manifest.get("crop_box_mm", {})
        x_min, x_max = crop.get("x_min", -np.inf), crop.get("x_max", np.inf)
        y_min, y_max = crop.get("y_min", -np.inf), crop.get("y_max", np.inf)
        z_min, z_max = crop.get("z_min", -np.inf), crop.get("z_max", np.inf)
        
        pixels_per_mm = manifest.get('pixels_per_mm', 1.0)
        mm_per_frame = manifest.get('mm_per_frame', 0.1)
        rot_deg = sweep_info.get('rotation_angle_deg', 0.0)
        rot_rad = np.deg2rad(rot_deg)
        
        # Build 2D mask: use manual quad if provided, otherwise project manifest crop box
        if override_mask_poly is not None:
            pixel_poly = np.array(override_mask_poly, dtype=np.int32)
            print(f"  Using manual quad mask ({len(pixel_poly)} vertices)")
        else:
            world_corners = np.array([
                [x_min, y_min], [x_max, y_min], [x_max, y_max], [x_min, y_max]
            ])
            cos_t, sin_t = np.cos(-rot_rad), np.sin(-rot_rad)
            cam_x = world_corners[:, 0] * cos_t - world_corners[:, 1] * sin_t
            cam_y = world_corners[:, 0] * sin_t + world_corners[:, 1] * cos_t
            u_corners = cam_x * pixels_per_mm + (width / 2.0)
            v_corners = cam_y * pixels_per_mm + (height / 2.0)
            pixel_poly = np.column_stack((u_corners, v_corners)).astype(np.int32)
        
        proj_mask = np.zeros((height, width), dtype=np.float32)
        cv2.fillPoly(proj_mask, [pixel_poly], 1.0)
        
        # Per-sweep adaptive background subtraction:
        # Sample frames evenly across the valid Z range, compute the temporal
        # median to capture static surfaces (floor, wall, table) that appear
        # at constant brightness regardless of Z position.
        # Web strands pass through the scan plane and don't dominate the median.
        n_bg_samples = 30
        first_valid_frame = max(start_frame, int(np.ceil(z_min / mm_per_frame)))
        last_valid_frame = min(start_frame + process_count - 1, int(np.floor(z_max / mm_per_frame)))
        
        if last_valid_frame > first_valid_frame:
            sample_indices = np.linspace(first_valid_frame, last_valid_frame, n_bg_samples, dtype=int)
            bg_samples = []
            for si in sample_indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, si)
                ret, frame = cap.read()
                if ret:
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
                    gray = gray * proj_mask
                    bg_samples.append(gray)
            
            if bg_samples:
                bg_median = np.median(np.array(bg_samples), axis=0)
                print(f"  Background model: median of {len(bg_samples)} samples, "
                      f"bg pixels above 0.05: {np.sum(bg_median > 0.05)}")
            else:
                bg_median = np.zeros((height, width), dtype=np.float32)
            
            # Reset video position for main loop
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame if start_frame > 0 else 0)
        else:
            bg_median = np.zeros((height, width), dtype=np.float32)
        
        all_detections = [np.zeros((0, 7))] * start_frame
        
        # Temporal frame stacking: average each frame with K neighbors
        # to boost SNR for faint but spatially consistent strands.
        K = self.config.temporal_stack_frames
        
        if K > 0:
            # Pre-read frames into buffer for sliding window
            from collections import deque
            
            # We need to read K frames ahead, so back up if possible
            actual_start = max(0, start_frame - K)
            if actual_start != start_frame:
                cap.set(cv2.CAP_PROP_POS_FRAMES, actual_start)
            
            # Read all needed frames into a buffer
            raw_grays = []
            for i in range(actual_start, min(start_frame + process_count + K, total_frames)):
                ret, frame = cap.read()
                if not ret:
                    break
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
                gray = gray * proj_mask
                gray = np.clip(gray - bg_median, 0, 1)  # Subtract static background
                raw_grays.append(gray)
            
            # Build stacked frames using sliding window mean
            offset = start_frame - actual_start  # index into raw_grays for start_frame
            
            print(f"Processing {video_path} (with {2*K+1}-frame temporal stacking)...")
            pbar = tqdm(total=process_count)
            
            for i in range(process_count):
                f = start_frame + i
                buf_idx = offset + i  # index of frame f in raw_grays
                
                z_curr = f * mm_per_frame
                if z_curr < z_min or z_curr > z_max:
                    all_detections.append(np.zeros((0, 7)))
                    pbar.update(1)
                    continue
                
                # Sliding window: average frames [buf_idx-K, buf_idx+K]
                win_start = max(0, buf_idx - K)
                win_end = min(len(raw_grays), buf_idx + K + 1)
                stacked = np.mean(raw_grays[win_start:win_end], axis=0).astype(np.float32)
                
                if self.config.method == 'steger':
                    dets = stegers_line_detection(
                        stacked, 
                        sigma=self.config.sigma, 
                        intensity_threshold=self.config.intensity_threshold,
                        strength_threshold=self.config.strength_threshold,
                        high_intensity_threshold=self.config.high_intensity_threshold,
                        high_strength_threshold=self.config.high_strength_threshold,
                        spatial_2d_min_length_px=self.config.spatial_2d_min_length_px
                    )
                else:
                    dets = simple_threshold_detection(
                        stacked, 
                        intensity_threshold=self.config.intensity_threshold
                    )
                all_detections.append(dets)
                pbar.update(1)
        else:
            print(f"Processing {video_path}...")
            pbar = tqdm(total=process_count)
            
            for f in range(start_frame, start_frame + process_count):
                ret, frame = cap.read()
                if not ret:
                    break
                    
                z_curr = f * mm_per_frame
                if z_curr < z_min or z_curr > z_max:
                    all_detections.append(np.zeros((0, 7)))
                    pbar.update(1)
                    continue
                    
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
                gray = gray * proj_mask
                gray = np.clip(gray - bg_median, 0, 1)  # Subtract static background
                if self.config.method == 'steger':
                    dets = stegers_line_detection(
                        gray, 
                        sigma=self.config.sigma, 
                        intensity_threshold=self.config.intensity_threshold,
                        strength_threshold=self.config.strength_threshold,
                        high_intensity_threshold=self.config.high_intensity_threshold,
                        high_strength_threshold=self.config.high_strength_threshold,
                        spatial_2d_min_length_px=self.config.spatial_2d_min_length_px
                    )
                else:
                    dets = simple_threshold_detection(
                        gray, 
                        intensity_threshold=self.config.intensity_threshold
                    )
                all_detections.append(dets)
                pbar.update(1)
            
        cap.release()
        pbar.close()
        
        print("Applying temporal coherence filter...")
        filtered_dets = temporal_coherence_filter(
            all_detections, 
            min_frames=self.config.persistence_min_frames,
            spatial_radius=self.config.temporal_spatial_radius,
            max_gap_frames=self.config.temporal_max_gap_frames
        )
        
        pcd_data = self._assemble_volume(filtered_dets, manifest, sweep_info, width, height)
        
        # Per-sweep adaptive spatial crop: trim to percentile-based bounding box.
        # The web occupies a similar XY extent from any angle. The wall/floor 
        # appears as extra points at the spatial periphery. Trimming to the 
        # 2nd-98th percentile of XY removes edge-region background without 
        # any geometric assumptions.
        if len(pcd_data) > 100:
            pcd_data = self._adaptive_spatial_crop(pcd_data)
        
        return pcd_data
    
    def _adaptive_spatial_crop(self, pcd_data, percentile_lo=2, percentile_hi=98):
        """
        Crop a sweep's point cloud to the robust bounding box derived from
        percentiles of the XY distribution. This trims away wall/floor points
        that extend beyond the web's core spatial extent.
        """
        x = pcd_data[:, 0]
        y = pcd_data[:, 1]
        
        x_lo, x_hi = np.percentile(x, percentile_lo), np.percentile(x, percentile_hi)
        y_lo, y_hi = np.percentile(y, percentile_lo), np.percentile(y, percentile_hi)
        
        keep = (x >= x_lo) & (x <= x_hi) & (y >= y_lo) & (y <= y_hi)
        
        n_removed = np.sum(~keep)
        print(f"  Adaptive crop: X=[{x_lo:.1f}, {x_hi:.1f}], Y=[{y_lo:.1f}, {y_hi:.1f}], "
              f"removed {n_removed} ({100*n_removed/len(pcd_data):.1f}%)")
        
        return pcd_data[keep]

    def _assemble_volume(self, filtered_detections, manifest, sweep_info, width, height):
        points = []
        
        pixels_per_mm = manifest.get('pixels_per_mm', 1.0)
        mm_per_frame = manifest.get('mm_per_frame', 0.1)
        rot_deg = sweep_info.get('rotation_angle_deg', 0.0)
        rot_rad = np.deg2rad(rot_deg)
        sweep_idx = sweep_info.get('id', 'unknown')
        
        principal_x = width / 2.0
        principal_y = height / 2.0
        
        numeric_sweep_id = 0.0
        
        for f, dets in enumerate(filtered_detections):
            if len(dets) == 0:
                continue
                
            u = dets[:, 0]
            v = dets[:, 1]
            intensities = dets[:, 2]
            strengths = dets[:, 3]
            pers = dets[:, 6]
            
            cam_y = (v - principal_y) / pixels_per_mm
            world_z = f * mm_per_frame
            cam_x = (u - principal_x) / pixels_per_mm
            
            world_x = cam_x * np.cos(rot_rad) - cam_y * np.sin(rot_rad)
            world_y = cam_x * np.sin(rot_rad) + cam_y * np.cos(rot_rad)
            z_rot = world_z
            
            for i in range(len(dets)):
                points.append([
                    world_x[i], world_y[i], z_rot,
                    numeric_sweep_id, f, u[i], v[i],
                    strengths[i], pers[i]
                ])
                
        return np.array(points) if points else np.zeros((0, 9))

class SweepMerger:
    """Merges multiple sweep PCDs into a single deduplicated PCD."""
    def __init__(self, config):
        self.config = config
        
    def merge_sweeps(self, pcds_data, manifest=None):
        print("Applying 3D polygon crop to individual sweeps...")
        cropped_pcds = []
        if manifest and "crop_box_mm" in manifest:
            crop = manifest["crop_box_mm"]
            x_min, x_max = crop.get("x_min", -np.inf), crop.get("x_max", np.inf)
            y_min, y_max = crop.get("y_min", -np.inf), crop.get("y_max", np.inf)
            z_min, z_max = crop.get("z_min", -np.inf), crop.get("z_max", np.inf)
            
            for pcd in pcds_data:
                if len(pcd) == 0:
                    cropped_pcds.append(pcd)
                    continue
                x, y, z = pcd[:, 0], pcd[:, 1], pcd[:, 2]
                valid = (x >= x_min) & (x <= x_max) & (y >= y_min) & (y <= y_max) & (z >= z_min) & (z <= z_max)
                cropped_pcds.append(pcd[valid])
                print(f"Points after crop: {np.sum(valid)}")
        else:
            cropped_pcds = pcds_data

        print("Registering sweeps...")
        registered_pcds = self._register_sweeps(cropped_pcds)
        
        all_points = np.vstack(registered_pcds) if registered_pcds else np.zeros((0, 9))
        
        print("Applying Adaptive Visual Hull Crop...")
        all_points = self._apply_visual_hull_crop(all_points, registered_pcds)
            
        print("Deduplicating union...")
        final_pcd = self._deduplicate_union(all_points, self.config.dedup_radius_mm)
        return final_pcd
        
    def _apply_visual_hull_crop(self, all_points, registered_pcds):
        if len(all_points) == 0 or len(registered_pcds) <= 1:
            return all_points
            
        hulls = []
        for pcd in registered_pcds:
            if len(pcd) < 3:
                continue
            xy = pcd[:, :2]
            try:
                hull = ConvexHull(xy)
                hull_points = xy[hull.vertices]
                delaunay = Delaunay(hull_points)
                hulls.append(delaunay)
            except Exception as e:
                print(f"Skipping hull generation for a sweep: {e}")
                
        if not hulls:
            return all_points
            
        keep = np.ones(len(all_points), dtype=bool)
        xy_all = all_points[:, :2]
        
        for delaunay in hulls:
            inside = delaunay.find_simplex(xy_all) >= 0
            keep = keep & inside
            
        print(f"Points remaining after Adaptive Visual Hull Crop: {np.sum(keep)}")
        return all_points[keep]

    def _register_sweeps(self, pcds_data):
        if not pcds_data or len(pcds_data) <= 1:
            return pcds_data
            
        registered = [pcds_data[0]]
        
        target_pts = pcds_data[0]
        if len(target_pts) == 0:
            return pcds_data
            
        target_pcd = o3d.geometry.PointCloud()
        target_pcd.points = o3d.utility.Vector3dVector(target_pts[:, :3])
        target_down = target_pcd.voxel_down_sample(self.config.icp_voxel_size)
        
        for i in range(1, len(pcds_data)):
            source_pts = pcds_data[i]
            if len(source_pts) == 0:
                registered.append(source_pts)
                continue
                
            source_pcd = o3d.geometry.PointCloud()
            source_pcd.points = o3d.utility.Vector3dVector(source_pts[:, :3])
            source_down = source_pcd.voxel_down_sample(self.config.icp_voxel_size)
            
            result_icp = o3d.pipelines.registration.registration_icp(
                source_down, target_down, self.config.icp_distance_threshold, np.eye(4),
                o3d.pipelines.registration.TransformationEstimationPointToPoint(),
                o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=50)
            )
            
            T = result_icp.transformation
            print(f"Sweep {i+1} ICP Fitness: {result_icp.fitness:.4f}")
            
            xyz = source_pts[:, :3]
            xyz_homog = np.column_stack((xyz, np.ones(len(xyz))))
            xyz_trans = (T @ xyz_homog.T).T[:, :3]
            
            source_pts_aligned = source_pts.copy()
            source_pts_aligned[:, :3] = xyz_trans
            
            registered.append(source_pts_aligned)
            
        return registered

    def _deduplicate_union(self, all_points, dedup_radius_mm):
        if len(all_points) == 0:
            return all_points
            
        xyz = all_points[:, :3]
        strengths = all_points[:, 7]
        tree = cKDTree(xyz)
        
        # Non-Maximum Suppression (NMS)
        # Sort indices by strength descending
        sorted_indices = np.argsort(-strengths)
        
        keep = np.zeros(len(all_points), dtype=bool)
        suppressed = np.zeros(len(all_points), dtype=bool)
        
        for i in sorted_indices:
            if suppressed[i]:
                continue
            keep[i] = True
            # Find all points within radius and suppress them
            neighbors = tree.query_ball_point(xyz[i], r=dedup_radius_mm)
            for n in neighbors:
                suppressed[n] = True
                    
        return all_points[keep]

def save_pcd(pcd_data, path):
    if len(pcd_data) == 0:
        print("Empty point cloud, not saving.")
        return
        
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pcd_data[:, :3])
    
    strengths = pcd_data[:, 7]
    norm_s = (strengths - strengths.min()) / (strengths.max() - strengths.min() + 1e-6)
    colors = np.column_stack((norm_s, norm_s, norm_s))
    pcd.colors = o3d.utility.Vector3dVector(colors)
    
    o3d.io.write_point_cloud(path, pcd)
    print(f"Saved {len(pcd.points)} points to {path}")
