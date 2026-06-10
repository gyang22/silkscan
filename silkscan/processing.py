import numpy as np
import cv2
import open3d as o3d
from tqdm import tqdm
from scipy.spatial import cKDTree, ConvexHull, Delaunay
from .line_detection import stegers_line_detection, simple_threshold_detection
from .filters import temporal_coherence_filter


class SweepProcessor:
    def __init__(self, config):
        self.config = config
        
    def process_video(self, video_path, sweep_info, manifest, start_frame=0, max_frames=None, override_mask_poly=None):
        cap = cv2.VideoCapture(video_path)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if start_frame > 0: cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        p_count = min(total_frames - start_frame, max_frames) if max_frames else total_frames - start_frame
        
        crop = manifest.get("crop_box_mm", {})
        ppm = manifest.get('pixels_per_mm', 1.0); mpf = manifest.get('mm_per_frame', 0.1)
        rot_rad = np.deg2rad(sweep_info.get('rotation_angle_deg', 0.0))
        
        if override_mask_poly is not None:
            pixel_poly = np.array(override_mask_poly, dtype=np.int32)
        else:
            xn, xx = crop.get("x_min", -1e6), crop.get("x_max", 1e6)
            yn, yx = crop.get("y_min", -1e6), crop.get("y_max", 1e6)
            wc = np.array([[xn, yn], [xx, yn], [xx, yx], [xn, yx]])
            ct, st = np.cos(-rot_rad), np.sin(-rot_rad)
            cx = wc[:, 0] * ct - wc[:, 1] * st; cy = wc[:, 0] * st + wc[:, 1] * ct
            pixel_poly = np.column_stack((cx * ppm + width/2.0, cy * ppm + height/2.0)).astype(np.int32)
        
        proj_mask = np.zeros((height, width), dtype=np.float32)
        cv2.fillPoly(proj_mask, [pixel_poly], 1.0)
        
        # Adaptive background subtraction
        z_min, z_max = crop.get("z_min", -1e6), crop.get("z_max", 1e6)
        f_start = max(start_frame, int(np.ceil(z_min / mpf))); f_end = min(start_frame + p_count - 1, int(np.floor(z_max / mpf)))
        if f_end > f_start:
            samples = []
            for si in np.linspace(f_start, f_end, 30, dtype=int):
                cap.set(cv2.CAP_PROP_POS_FRAMES, si)
                ret, frame = cap.read()
                if ret: samples.append(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32)/255.0 * proj_mask)
            bg_median = np.median(np.array(samples), axis=0) if samples else np.zeros((height, width), dtype=np.float32)
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        else: bg_median = np.zeros((height, width), dtype=np.float32)
        
        all_dets = [np.zeros((0, 7))] * start_frame
        K = self.config.temporal_stack_frames
        if K > 0:
            a_start = max(0, start_frame - K)
            cap.set(cv2.CAP_PROP_POS_FRAMES, a_start)
            raw_grays = []
            for i in range(a_start, min(start_frame + p_count + K, total_frames)):
                ret, frame = cap.read()
                if not ret: break
                raw_grays.append(np.clip(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32)/255.0 * proj_mask - bg_median, 0, 1))
            offset = start_frame - a_start
            for i in tqdm(range(p_count), desc=f"Stacking {video_path}"):
                f = start_frame + i; buf_idx = offset + i
                z_curr = f * mpf
                if z_curr < z_min or z_curr > z_max:
                    all_dets.append(np.zeros((0, 7))); continue
                stacked = np.mean(raw_grays[max(0, buf_idx-K):min(len(raw_grays), buf_idx+K+1)], axis=0).astype(np.float32)
                all_dets.append(self._detect(stacked))
        else:
            for f in tqdm(range(start_frame, start_frame + p_count), desc=f"Processing {video_path}"):
                ret, frame = cap.read()
                if not ret: break
                z_curr = f * mpf
                if z_curr < z_min or z_curr > z_max:
                    all_dets.append(np.zeros((0, 7))); continue
                gray = np.clip(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32)/255.0 * proj_mask - bg_median, 0, 1)
                all_dets.append(self._detect(gray))
        cap.release()
        if self.config.method == 'steger':
            filtered = temporal_coherence_filter(all_dets, min_frames=self.config.persistence_min_frames, spatial_radius=self.config.temporal_spatial_radius, max_gap_frames=self.config.temporal_max_gap_frames)
        else:
            filtered = all_dets
            
        pcd = self._assemble(filtered, manifest, sweep_info, width, height)
        return self._adaptive_spatial_crop(pcd) if len(pcd) > 100 else pcd

    def _detect(self, img):
        if self.config.method == 'steger':
            return stegers_line_detection(img, sigma=self.config.sigma, intensity_threshold=self.config.intensity_threshold, strength_threshold=self.config.strength_threshold, high_intensity_threshold=self.config.high_intensity_threshold, high_strength_threshold=self.config.high_strength_threshold, spatial_2d_min_length_px=self.config.spatial_2d_min_length_px)
        return simple_threshold_detection(img, intensity_threshold=self.config.intensity_threshold)

    def _adaptive_spatial_crop(self, pcd, p_lo=2, p_hi=98):
        x, y = pcd[:, 0], pcd[:, 1]
        x_lo, x_hi = np.percentile(x, p_lo), np.percentile(x, p_hi)
        y_lo, y_hi = np.percentile(y, p_lo), np.percentile(y, p_hi)
        keep = (x >= x_lo) & (x <= x_hi) & (y >= y_lo) & (y <= y_hi)
        return pcd[keep]

    def _assemble(self, detections, manifest, sweep_info, width, height):
        pts = []; ppm = manifest.get('pixels_per_mm', 1.0); mpf = manifest.get('mm_per_frame', 0.1)
        rot_rad = np.deg2rad(sweep_info.get('rotation_angle_deg', 0.0))
        px, py = width/2.0, height/2.0
        for f, dets in enumerate(detections):
            if len(dets) == 0: continue
            u, v, ints, strs, _, _, pers = dets.T
            cam_y = (v - py) / ppm; cam_x = (u - px) / ppm; world_z = f * mpf
            wx = cam_x * np.cos(rot_rad) - cam_y * np.sin(rot_rad); wy = cam_x * np.sin(rot_rad) + cam_y * np.cos(rot_rad)
            for i in range(len(dets)): pts.append([wx[i], wy[i], world_z, ints[i], f, u[i], v[i], strs[i], pers[i]])
        return np.array(pts) if pts else np.zeros((0, 9))

class SweepMerger:
    def __init__(self, config):
        self.config = config
    def merge_sweeps(self, pcds, manifest=None):
        cropped = []
        if manifest and "crop_box_mm" in manifest:
            c = manifest["crop_box_mm"]; xn, xx, yn, yx, zn, zx = c.get("x_min", -np.inf), c.get("x_max", np.inf), c.get("y_min", -np.inf), c.get("y_max", np.inf), c.get("z_min", -np.inf), c.get("z_max", np.inf)
            for p in pcds:
                if len(p) == 0: cropped.append(p); continue
                v = (p[:,0]>=xn)&(p[:,0]<=xx)&(p[:,1]>=yn)&(p[:,1]<=yx)&(p[:,2]>=zn)&(p[:,2]<=zx); cropped.append(p[v])
        else: cropped = pcds
        reg = self._register(cropped); all_p = np.vstack(reg) if reg else np.zeros((0, 9))
        all_p = self._visual_hull_crop(all_p, reg)
        return self._dedup(all_p, self.config.dedup_radius_mm)

    def _visual_hull_crop(self, all_pts, reg):
        if len(all_pts) == 0 or len(reg) <= 1: return all_pts
        hulls = []
        for p in reg:
            if len(p) < 3: continue
            try: hulls.append(Delaunay(p[ConvexHull(p[:,:2]).vertices, :2]))
            except: pass
        if not hulls: return all_pts
        keep = np.ones(len(all_pts), dtype=bool)
        for d in hulls: keep &= (d.find_simplex(all_pts[:,:2]) >= 0)
        return all_pts[keep]

    def _register(self, pcds):
        if not pcds or len(pcds) <= 1: return pcds
        reg = [pcds[0]]; target_pcd = o3d.geometry.PointCloud(); target_pcd.points = o3d.utility.Vector3dVector(pcds[0][:, :3])
        target_down = target_pcd.voxel_down_sample(self.config.icp_voxel_size)
        for i in range(1, len(pcds)):
            if len(pcds[i]) == 0: reg.append(pcds[i]); continue
            s_pcd = o3d.geometry.PointCloud(); s_pcd.points = o3d.utility.Vector3dVector(pcds[i][:, :3])
            s_down = s_pcd.voxel_down_sample(self.config.icp_voxel_size)
            res = o3d.pipelines.registration.registration_icp(s_down, target_down, self.config.icp_distance_threshold, np.eye(4), o3d.pipelines.registration.TransformationEstimationPointToPoint())
            xyz = pcds[i][:, :3]; xyz_h = np.column_stack((xyz, np.ones(len(xyz))))
            xyz_t = (res.transformation @ xyz_h.T).T[:, :3]; p_aligned = pcds[i].copy(); p_aligned[:, :3] = xyz_t; reg.append(p_aligned)
        return reg

    def _dedup(self, pts, radius):
        if len(pts) == 0: return pts
        tree = cKDTree(pts[:, :3]); s_idx = np.argsort(-pts[:, 7]); keep = np.zeros(len(pts), dtype=bool); suppressed = np.zeros(len(pts), dtype=bool)
        for i in s_idx:
            if suppressed[i]: continue
            keep[i] = True; neighbors = tree.query_ball_point(pts[i, :3], r=radius)
            for n in neighbors: suppressed[n] = True
        return pts[keep]
