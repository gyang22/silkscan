import numpy as np

def save_pcd(pcd_data, path):
    """Write a point cloud to a binary PCD file.

    Deliberately avoids Open3D: this function runs inside the GUI's background
    processing thread, and constructing Open3D (pybind11) objects off the main
    thread segfaults non-deterministically on some macOS / library builds.
    Writing the PCD directly is version-independent and thread-safe. Open3D is
    only used in the separate editor subprocess, on that process's main thread.
    """
    if len(pcd_data) == 0:
        print("Empty point cloud, not saving.")
        return

    xyz = np.ascontiguousarray(pcd_data[:, :3], dtype=np.float32)
    n = len(xyz)

    has_color = pcd_data.shape[1] >= 8
    if has_color:
        # Pack normalized strength as grayscale rgb, matching PCL/Open3D's
        # single-float 0x00RRGGBB encoding so the file reads back identically.
        strengths = pcd_data[:, 7].astype(np.float64)
        denom = strengths.max() - strengths.min() + 1e-6
        gray = np.clip((strengths - strengths.min()) / denom * 255.0, 0, 255).astype(np.uint32)
        rgb_int = (gray << 16) | (gray << 8) | gray
        rgb_f = rgb_int.view(np.float32)

        record = np.zeros(n, dtype=[('x', '<f4'), ('y', '<f4'), ('z', '<f4'), ('rgb', '<f4')])
        record['rgb'] = rgb_f
        fields, sizes, types, counts = "x y z rgb", "4 4 4 4", "F F F F", "1 1 1 1"
    else:
        record = np.zeros(n, dtype=[('x', '<f4'), ('y', '<f4'), ('z', '<f4')])
        fields, sizes, types, counts = "x y z", "4 4 4", "F F F", "1 1 1"

    record['x'] = xyz[:, 0]
    record['y'] = xyz[:, 1]
    record['z'] = xyz[:, 2]

    header = (
        "# .PCD v0.7 - Point Cloud Data file format\n"
        "VERSION 0.7\n"
        f"FIELDS {fields}\n"
        f"SIZE {sizes}\n"
        f"TYPE {types}\n"
        f"COUNT {counts}\n"
        f"WIDTH {n}\n"
        "HEIGHT 1\n"
        "VIEWPOINT 0 0 0 1 0 0 0\n"
        f"POINTS {n}\n"
        "DATA binary\n"
    )

    with open(path, 'wb') as f:
        f.write(header.encode('ascii'))
        f.write(record.tobytes())

    print(f"Saved {n} points to {path}")
