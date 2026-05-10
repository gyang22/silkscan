import os
import cv2
import numpy as np
from tqdm import tqdm

def get_brightness_cached(sweep_id, vid_path, cache_dir, step=5):
    """
    Computes per-frame brightness and caches it as a .npy file.
    """
    cache_file = os.path.join(cache_dir, f'{sweep_id}_brightness_s{step}.npy')
    idx_file = os.path.join(cache_dir, f'{sweep_id}_frame_idx_s{step}.npy')
    
    if os.path.exists(cache_file) and os.path.exists(idx_file):
        return np.load(idx_file), np.load(cache_file)
    
    cap = cv2.VideoCapture(vid_path)
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_idx = np.array(list(range(0, n_frames, step)))
    brightness = np.zeros(len(frame_idx))
    
    print(f'  Computing brightness for {sweep_id} ({n_frames} frames)...')
    for bi, f in enumerate(tqdm(frame_idx)):
        cap.set(cv2.CAP_PROP_POS_FRAMES, f)
        ret, frame = cap.read()
        if not ret: break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float64)/255.0
        brightness[bi] = gray.sum()
    cap.release()
    
    np.save(cache_file, brightness)
    np.save(idx_file, frame_idx)
    return frame_idx, brightness

def get_sum_image_cached(sweep_id, vid_path, start_frame, end_frame, cache_dir, step=5):
    """
    Computes temporal sum of frames and caches it.
    The cache is keyed only by sweep_id and step to be less sensitive to 
    slight variations in the detected start/end frames.
    """
    cache_file = os.path.join(cache_dir, f'{sweep_id}_sum_s{step}.npy')
    
    if os.path.exists(cache_file):
        return np.load(cache_file)

    
    cap = cv2.VideoCapture(vid_path)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    accum = np.zeros((h, w), dtype=np.float64)
    frames = list(range(start_frame, end_frame+1, step))
    
    print(f'  Summing {len(frames)} frames for {sweep_id}...')
    for f in tqdm(frames):
        cap.set(cv2.CAP_PROP_POS_FRAMES, f)
        ret, frame = cap.read()
        if not ret: break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float64)/255.0
        accum += gray
    cap.release()
    
    np.save(cache_file, accum)
    return accum
