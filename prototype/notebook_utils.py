import matplotlib.pyplot as plt
import numpy as np
import cv2

def plot_diagnostic_row(si, sweep_info, result_dict, axes, accum, sweep_info_manifest, w_img, h_img):
    """
    Plots a column of diagnostic visualizations for a single sweep.
    """
    sid = sweep_info['id']
    
    # Row 1-2 plotting is handled in the main loop for convenience, 
    # but Row 3-5 can be helper-ized.
    
    # Row 3: Rotation Search
    ax = axes[2, si]
    ax.plot(result_dict['angles'], result_dict['scores'], 'b.-', lw=1, label='Coarse')
    ax.plot(result_dict['fine_angles'], result_dict['fine_scores'], 'r.-', lw=1, label='Fine')
    ax.axvline(result_dict['best_angle'], color='cyan', lw=2, label=f"Best={result_dict['best_angle']:.1f}deg")
    ax.set_xlabel('Angle (deg)'); ax.set_ylabel('Sharpness')
    ax.legend(fontsize=7)
    ax.set_title(f'{sid} rotation search')
    
    # Row 4: Projections
    ax = axes[3, si]
    col_proj = result_dict['col_proj']
    row_proj = result_dict['row_proj']
    ax.plot(col_proj/col_proj.max(), 'b-', lw=0.8, alpha=0.7, label='Col')
    ax.plot(row_proj/row_proj.max(), 'g-', lw=0.8, alpha=0.7, label='Row')
    ax.axvline(result_dict['xl'], color='cyan', lw=2, ls='--', label=f"xL={result_dict['xl']}")
    ax.axvline(result_dict['xr'], color='cyan', lw=2, ls='-', label=f"xR={result_dict['xr']}")
    ax.axvline(result_dict['yt'], color='yellow', lw=2, ls='--', label=f"yT={result_dict['yt']}")
    ax.axvline(result_dict['yb'], color='yellow', lw=2, ls='-', label=f"yB={result_dict['yb']}")
    ax.legend(fontsize=6, loc='upper left')
    ax.set_title(f"{sid} projections @ {result_dict['best_angle']:.1f}deg")
    
    # Row 5: Quad Overlay
    ax = axes[4, si]
    ax.imshow(accum, cmap='hot', aspect='auto')
    quad = result_dict['quad']
    qc = np.vstack([quad, quad[0:1]])
    ax.plot(qc[:,0], qc[:,1], 'c-', lw=3, label='Detected quad')
    ax.scatter(quad[:,0], quad[:,1], c='cyan', s=120, zorder=5)
    
    # Manifest comparison
    crop = sweep_info_manifest.get('crop_box_mm', {})
    xn, xx = crop.get('x_min', -48), crop.get('x_max', 48)
    yn, yx = crop.get('y_min', -48), crop.get('y_max', 48)
    ppm = sweep_info_manifest.get('pixels_per_mm', 3.4)
    rot = np.deg2rad(sweep_info.get('rotation_angle_deg', 0.0))
    wc = np.array([[xn,yn],[xx,yn],[xx,yx],[xn,yx]])
    ct, st = np.cos(-rot), np.sin(-rot)
    cu = (wc[:,0]*ct - wc[:,1]*st)*ppm + w_img/2.0
    cv_ = (wc[:,0]*st + wc[:,1]*ct)*ppm + h_img/2.0
    mp = np.column_stack((cu, cv_))
    mpc = np.vstack([mp, mp[0:1]])
    ax.plot(mpc[:,0], mpc[:,1], 'g--', lw=2, label='Manifest')
    ax.legend(fontsize=9)
    ax.set_title(f"{sid} quad vs manifest")
