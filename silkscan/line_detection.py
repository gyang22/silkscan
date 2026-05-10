import numpy as np
import cv2
from skimage.feature import hessian_matrix, hessian_matrix_eigvals
from scipy.ndimage import gaussian_filter1d
from scipy.spatial import cKDTree
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

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
        c_idx, r_idx, intensities, strengths, n_c, n_r, is_high_conf
    ))

def stegers_line_detection(image, 
                           sigma=0.5, 
                           intensity_threshold=0.05, 
                           strength_threshold=0.002, 
                           high_intensity_threshold=0.15, 
                           high_strength_threshold=0.015,
                           spatial_2d_min_length_px=20):
    mask = (image >= intensity_threshold)
    r_idx, c_idx = np.where(mask)
    if len(r_idx) == 0:
        return np.zeros((0, 7))
        
    intensities = image[r_idx, c_idx]
    is_bright = intensities >= high_intensity_threshold
    H_elems = hessian_matrix(image, sigma=sigma, order='rc')
    eigvals = hessian_matrix_eigvals(H_elems)
    lambda_min = eigvals[1]
    H_rr, H_rc, H_cc = H_elems
    
    v_r = -H_rc[r_idx, c_idx]
    v_c = H_rr[r_idx, c_idx] - lambda_min[r_idx, c_idx]
    norm = np.sqrt(v_r**2 + v_c**2)
    norm[norm == 0] = 1.0
    n_r = v_r / norm
    n_c = v_c / norm
    
    grad_r = gaussian_filter1d(image, sigma=sigma, axis=0, order=1)
    grad_r = gaussian_filter1d(grad_r, sigma=sigma, axis=1, order=0)
    grad_c = gaussian_filter1d(image, sigma=sigma, axis=1, order=1)
    grad_c = gaussian_filter1d(grad_c, sigma=sigma, axis=0, order=0)
    
    g_r, g_c = grad_r[r_idx, c_idx], grad_c[r_idx, c_idx]
    g_dot_n = g_r * n_r + g_c * n_c
    denom = lambda_min[r_idx, c_idx]
    t = np.zeros_like(denom)
    valid_denom = np.abs(denom) > 1e-6
    t[valid_denom] = - g_dot_n[valid_denom] / denom[valid_denom]
    
    is_ridge = (np.abs(t) <= 0.5) & (denom < 0)
    keep = is_bright | (is_ridge & ~is_bright)
    
    r_idx = r_idx[keep]; c_idx = c_idx[keep]
    n_r = n_r[keep]; n_c = n_c[keep]
    t_keep = t[keep]
    t_keep[is_bright[keep]] = np.clip(t_keep[is_bright[keep]], -0.5, 0.5)
    
    if len(r_idx) == 0: return np.zeros((0, 7))
    
    r_sub = r_idx + t_keep * n_r
    c_sub = c_idx + t_keep * n_c
    
    intensities = image[r_idx, c_idx]
    strengths = np.abs(lambda_min[r_idx, c_idx])
    is_faint = intensities < high_intensity_threshold
    valid = ~is_faint | (strengths >= strength_threshold)
    
    pts = np.column_stack((c_sub[valid], r_sub[valid]))
    v_ints = intensities[valid]; v_strs = strengths[valid]
    n_c_v = n_c[valid]; n_r_v = n_r[valid]
    
    if spatial_2d_min_length_px > 1 and len(pts) > 0:
        faint_mask = v_ints < high_intensity_threshold
        if np.any(faint_mask):
            f_pts = pts[faint_mask]
            tree = cKDTree(f_pts)
            pairs = tree.query_pairs(r=1.5)
            if pairs:
                p_arr = np.array(list(pairs))
                adj = csr_matrix((np.ones(len(p_arr), dtype=bool), (p_arr[:, 0], p_arr[:, 1])), shape=(len(f_pts), len(f_pts)))
                _, labels = connected_components(csgraph=adj, directed=False)
                unique_labels, counts = np.unique(labels, return_counts=True)
                valid_labels = unique_labels[counts >= spatial_2d_min_length_px]
                f_keep = np.isin(labels, valid_labels)
            else:
                f_keep = np.zeros(np.sum(faint_mask), dtype=bool)
            
            final_keep = np.ones(len(pts), dtype=bool)
            final_keep[faint_mask] = f_keep
            pts = pts[final_keep]; v_ints = v_ints[final_keep]; v_strs = v_strs[final_keep]
            n_c_v = n_c_v[final_keep]; n_r_v = n_r_v[final_keep]
            
    if len(pts) == 0: return np.zeros((0, 7))
    is_high = (v_ints >= high_intensity_threshold) & (v_strs >= high_strength_threshold)
    return np.column_stack((pts[:, 0], pts[:, 1], v_ints, v_strs, n_c_v, n_r_v, is_high.astype(np.float32)))
