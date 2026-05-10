import os
import sys

# Add parent directory to path to find the 'silkscan' package if not installed
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if root_dir not in sys.path:
    sys.path.append(root_dir)

from silkscan import (
    CaptureSet, Config, load_capture_set,
    simple_threshold_detection, stegers_line_detection,
    temporal_coherence_filter,
    SweepProcessor, SweepMerger,
    save_pcd
)


# Maintain backward compatibility for any direct imports from silkscan_proto
__all__ = [
    'CaptureSet', 'Config', 'load_capture_set',
    'simple_threshold_detection', 'stegers_line_detection',
    'temporal_coherence_filter',
    'SweepProcessor', 'SweepMerger',
    'save_pcd'
]

