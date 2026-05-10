import numpy as np
from scipy.spatial import cKDTree
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from tqdm import tqdm
import itertools

def temporal_coherence_filter(frames_detections, min_frames=2, spatial_radius=2.0, max_gap_frames=1):
    if not frames_detections: return []
    num_frames = len(frames_detections)
    trees = []; kept = []
    
    for dets in frames_detections:
        if len(dets) > 0:
            trees.append(cKDTree(dets[:, :2]))
            kept.append(dets[:, 6] > 0.5)
        else:
            trees.append(None); kept.append(np.array([], dtype=bool))
            
    # Forward Pass
    for f in tqdm(range(num_frames), desc="Forward Propagation"):
        if not np.any(kept[f]): continue
        active_pts = frames_detections[f][kept[f], :2]
        for k in range(1, max_gap_frames + 1):
            n_f = f + k
            if n_f >= num_frames or trees[n_f] is None: continue
            idxs = trees[n_f].query_ball_point(active_pts, r=spatial_radius * k)
            if idxs:
                flat = np.fromiter(itertools.chain.from_iterable(idxs), dtype=int)
                if len(flat) > 0: kept[n_f][flat] = True
                    
    # Backward Pass
    for f in tqdm(range(num_frames - 1, -1, -1), desc="Backward Propagation"):
        if not np.any(kept[f]): continue
        active_pts = frames_detections[f][kept[f], :2]
        for k in range(1, max_gap_frames + 1):
            n_f = f - k
            if n_f < 0 or trees[n_f] is None: continue
            idxs = trees[n_f].query_ball_point(active_pts, r=spatial_radius * k)
            if idxs:
                flat = np.fromiter(itertools.chain.from_iterable(idxs), dtype=int)
                if len(flat) > 0: kept[n_f][flat] = True
                    
    # Component Persistence
    surviving_trees = []; global_offsets = np.zeros(num_frames, dtype=int)
    node_frames = []; node_orig_idx = []; current_offset = 0
    for f in range(num_frames):
        global_offsets[f] = current_offset
        if not np.any(kept[f]):
            surviving_trees.append(None); continue
        s_idx = np.where(kept[f])[0]
        s_pts = frames_detections[f][s_idx, :2]
        surviving_trees.append(cKDTree(s_pts))
        node_frames.append(np.full(len(s_idx), f))
        node_orig_idx.append(s_idx)
        current_offset += len(s_idx)
        
    num_nodes = current_offset
    if num_nodes == 0: return [np.zeros((0, 7)) for _ in range(num_frames)]
    node_frames = np.concatenate(node_frames); node_orig_idx = np.concatenate(node_orig_idx)
    
    I = []; J = []
    for f in tqdm(range(num_frames - 1), desc="Building Graph"):
        if surviving_trees[f] is None: continue
        s_pts_f = frames_detections[f][np.where(kept[f])[0], :2]
        for k in range(1, max_gap_frames + 1):
            n_f = f + k
            if n_f >= num_frames or surviving_trees[n_f] is None: continue
            n_pts_nf = np.sum(kept[n_f])
            dists, idxs = surviving_trees[n_f].query(s_pts_f, k=3, distance_upper_bound=spatial_radius * k)
            if idxs.ndim == 1: idxs = idxs.reshape(-1, 1)
            valid = idxs < n_pts_nf
            if not np.any(valid): continue
            u_global = global_offsets[f] + np.repeat(np.arange(len(s_pts_f)), idxs.shape[1]).reshape(idxs.shape)[valid]
            v_global = global_offsets[n_f] + idxs[valid]
            I.append(u_global); J.append(v_global)
            
    if I:
        adj = csr_matrix((np.ones(len(np.concatenate(I)), dtype=bool), (np.concatenate(I), np.concatenate(J))), shape=(num_nodes, num_nodes))
        n_comp, labels = connected_components(csgraph=adj, directed=False, return_labels=True)
    else:
        labels = np.arange(num_nodes); n_comp = num_nodes
                    
    unique_pairs = np.unique(np.column_stack((labels, node_frames)), axis=0)
    u_labels, counts = np.unique(unique_pairs[:, 0], return_counts=True)
    pers_map = np.zeros(n_comp, dtype=int)
    pers_map[u_labels] = counts
    
    keep_node_mask = pers_map[labels] >= min_frames
    filtered = []
    for f in range(num_frames):
        mask_f = (node_frames == f) & keep_node_mask
        if not np.any(mask_f):
            filtered.append(np.zeros((0, 7))); continue
        idx_to_keep = node_orig_idx[mask_f]
        filtered.append(np.column_stack((frames_detections[f][idx_to_keep, :6], pers_map[labels][mask_f])))
    return filtered
