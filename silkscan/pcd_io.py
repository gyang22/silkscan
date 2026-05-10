import open3d as o3d
import numpy as np

def save_pcd(pcd_data, path):
    if len(pcd_data) == 0:
        print("Empty point cloud, not saving.")
        return
        
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pcd_data[:, :3])
    
    strengths = pcd_data[:, 7]
    denom = strengths.max() - strengths.min() + 1e-6
    norm_s = (strengths - strengths.min()) / denom
    colors = np.column_stack((norm_s, norm_s, norm_s))
    pcd.colors = o3d.utility.Vector3dVector(colors)
    
    o3d.io.write_point_cloud(path, pcd)
    print(f"Saved {len(pcd.points)} points to {path}")
