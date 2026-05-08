import sys
import open3d as o3d
import argparse

def main():
    parser = argparse.ArgumentParser(description="Visualize a Point Cloud Data (.pcd) file.")
    parser.add_argument("pcd_file", help="Path to the .pcd file to visualize")
    args = parser.parse_args()

    print(f"Loading {args.pcd_file}...")
    try:
        pcd = o3d.io.read_point_cloud(args.pcd_file)
        if pcd.is_empty():
            print("Error: The point cloud is empty or the file could not be read properly.")
            sys.exit(1)
            
        print(f"Successfully loaded {len(pcd.points)} points.")
        print("Opening visualization window... (Press 'Q' or 'Esc' in the window to close)")
        
        # We can also add a coordinate frame for better spatial context
        coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=10.0, origin=[0, 0, 0])
        
        o3d.visualization.draw_geometries([pcd, coordinate_frame])
        
    except Exception as e:
        print(f"Failed to load or visualize the point cloud: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
