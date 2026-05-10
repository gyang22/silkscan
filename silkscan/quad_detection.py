import cv2
import numpy as np
from scipy.ndimage import uniform_filter1d

def detect_quad_rotated(accum, pad=20):
    """
    Detects a rotated rectangle boundaries by searching for peak intensities 
    in projection profiles across multiple rotation angles.
    """
    h_img, w_img = accum.shape
    cx, cy = w_img / 2.0, h_img / 2.0
    ks = 31
    margin_frac = 0.05

    def score_angle(angle_deg):
        M = cv2.getRotationMatrix2D((cx, cy), angle_deg, 1.0)
        rotated = cv2.warpAffine(accum.astype(np.float32), M, (w_img, h_img))
        
        col_proj = rotated.sum(axis=0)
        row_proj = rotated.sum(axis=1)
        
        # We use raw projections for peak finding as per user preference
        min_sep_x = int(w_img * 0.1)
        min_sep_y = int(h_img * 0.1)
        
        # X edges: find top 2 absolute intensity peaks
        p1x = np.argmax(col_proj)
        cp_tmp = col_proj.copy()
        cp_tmp[max(0, p1x - min_sep_x):min(len(cp_tmp), p1x + min_sep_x)] = 0
        p2x = np.argmax(cp_tmp)
        xl, xr = min(p1x, p2x), max(p1x, p2x)
        
        # Y edges: find top 2 absolute intensity peaks
        p1y = np.argmax(row_proj)
        rp_tmp = row_proj.copy()
        rp_tmp[max(0, p1y - min_sep_y):min(len(rp_tmp), p1y + min_sep_y)] = 0
        p2y = np.argmax(rp_tmp)
        yt, yb = min(p1y, p2y), max(p1y, p2y)
        
        sharpness = col_proj[p1x] + cp_tmp[p2x] + row_proj[p1y] + rp_tmp[p2y]
        
        # Calculate gradients for plotting/debugging purposes
        col_s = uniform_filter1d(col_proj, size=ks)
        row_s = uniform_filter1d(row_proj, size=ks)
        col_g = np.gradient(col_s)
        row_g = np.gradient(row_s)
        
        return sharpness, xl, xr, yt, yb, col_proj, row_proj, col_g, row_g

    # Search for best rotation angle
    angles = np.arange(-20, 21, 1.0)
    scores = np.array([score_angle(a)[0] for a in angles])
    best_coarse = angles[np.argmax(scores)]
    
    fine_angles = np.arange(best_coarse - 2, best_coarse + 2.1, 0.1)
    fine_scores = np.array([score_angle(a)[0] for a in fine_angles])
    best_angle = fine_angles[np.argmax(fine_scores)]
    
    # Get final boundaries at best angle
    _, xl, xr, yt, yb, col_proj, row_proj, col_g, row_g = score_angle(best_angle)
    
    # Apply padding (inset)
    xl += pad; xr -= pad; yt += pad; yb -= pad
    
    # Transform back to original image space
    rect_rot = np.array([[xl, yt], [xr, yt], [xr, yb], [xl, yb]], dtype=np.float64)
    theta = np.deg2rad(best_angle)
    cos_t, sin_t = np.cos(theta), np.sin(theta)
    centered = rect_rot - np.array([cx, cy])
    rotated_back = np.column_stack([
        centered[:, 0] * cos_t - centered[:, 1] * sin_t,
        centered[:, 0] * sin_t + centered[:, 1] * cos_t
    ]) + np.array([cx, cy])
    
    return {
        'quad': rotated_back.astype(np.float32),
        'best_angle': best_angle,
        'angles': angles,
        'scores': scores,
        'fine_angles': fine_angles,
        'fine_scores': fine_scores,
        'col_proj': col_proj,
        'row_proj': row_proj,
        'col_g': col_g,
        'row_g': row_g,
        'xl': xl, 'xr': xr, 'yt': yt, 'yb': yb
    }
