use pyo3::prelude::*;
use numpy::{PyArray2, PyArray3, PyReadonlyArray2, PyArrayMethods};
use kiddo::{KdTree, SquaredEuclidean, NearestNeighbour};
use rayon::prelude::*;
use std::collections::HashSet;
use std::process::{Command, Stdio};
use std::io::Read;

// Optimized Steger Detection for a single frame
fn steger_detect_frame(
    img: &[f32],
    width: usize,
    height: usize,
    sigma: f32,
    intensity_threshold: f32,
    strength_threshold: f32,
) -> Vec<[f32; 7]> {
    let k_radius = (sigma * 3.0).ceil() as usize;
    let k_size = k_radius * 2 + 1;
    let mut g = vec![0.0f32; k_size];
    let mut g1 = vec![0.0f32; k_size];
    let mut g2 = vec![0.0f32; k_size];
    
    let sigma2 = sigma * sigma;
    let sigma4 = sigma2 * sigma2;
    
    for i in 0..k_size {
        let x = i as f32 - k_radius as f32;
        let x2 = x * x;
        let base = (-x2 / (2.0 * sigma2)).exp();
        g[i] = base / (2.0 * std::f32::consts::PI * sigma2).sqrt();
        g1[i] = -x / sigma2 * g[i];
        g2[i] = (x2 - sigma2) / sigma4 * g[i];
    }
    let g_sum: f32 = g.iter().sum();
    for x in &mut g { *x /= g_sum; }

    // Use a single pre-allocated buffer for derivatives to avoid frequent allocations
    // However, in a parallel map, we'll allocate per-frame for now but make them more efficient.
    
    fn deriv_2d(data: &[f32], w: usize, h: usize, k_h: &[f32], k_v: &[f32]) -> Vec<f32> {
        let mut temp = vec![0.0; w * h];
        let mut res = vec![0.0; w * h];
        let k_half = k_h.len() / 2;
        for r in 0..h {
            let row_offset = r * w;
            for c in 0..w {
                let mut s = 0.0;
                for k in 0..k_h.len() {
                    let cc = (c as isize + k as isize - k_half as isize).clamp(0, w as isize - 1) as usize;
                    s += data[row_offset + cc] * k_h[k];
                }
                temp[row_offset + c] = s;
            }
        }
        for r in 0..h {
            for c in 0..w {
                let mut s = 0.0;
                for k in 0..k_v.len() {
                    let rr = (r as isize + k as isize - k_half as isize).clamp(0, h as isize - 1) as usize;
                    s += temp[rr * w + c] * k_v[k];
                }
                res[r * w + c] = s;
            }
        }
        res
    }

    let i_x = deriv_2d(img, width, height, &g1, &g);
    let i_y = deriv_2d(img, width, height, &g, &g1);
    let i_xx = deriv_2d(img, width, height, &g2, &g);
    let i_yy = deriv_2d(img, width, height, &g, &g2);
    let i_xy = deriv_2d(img, width, height, &g1, &g1);

    let mut detections = Vec::new();
    for r in 0..height {
        let offset = r * width;
        for c in 0..width {
            let idx = offset + c;
            let intensity = img[idx];
            if intensity < intensity_threshold { continue; }

            let lxx = i_xx[idx];
            let lxy = i_xy[idx];
            let lyy = i_yy[idx];

            let tr = lxx + lyy;
            let det = lxx * lyy - lxy * lxy;
            let disc = (tr * tr - 4.0 * det).max(0.0).sqrt();
            let lam1 = (tr - disc) / 2.0; 

            let strength = lam1.abs();
            if strength < strength_threshold { continue; }

            let mut nx = -lxy;
            let mut ny = lxx - lam1;
            let norm = (nx * nx + ny * ny).sqrt();
            if norm > 1e-9 { nx /= norm; ny /= norm; } else { continue; }

            let numer = i_x[idx] * nx + i_y[idx] * ny;
            let denom = lxx * nx * nx + 2.0 * lxy * nx * ny + lyy * ny * ny;
            if denom.abs() < 1e-6 { continue; }
            let t = -numer / denom;

            if t.abs() <= 0.5 {
                detections.push([c as f32 + t * nx, r as f32 + t * ny, intensity, strength, nx, ny, 1.0]);
            }
        }
    }
    detections
}

#[pyfunction]
#[pyo3(signature = (vid_path, start_frame, count, mask, bg_median, k_radius=0, sigma=0.5, intensity_threshold=0.05, strength_threshold=0.002))]
fn process_video_steger(
    py: Python<'_>,
    vid_path: String,
    start_frame: usize,
    count: usize,
    mask: PyReadonlyArray2<f32>,
    bg_median: PyReadonlyArray2<f32>,
    k_radius: usize,
    sigma: f32,
    intensity_threshold: f32,
    strength_threshold: f32,
) -> PyResult<Vec<Option<PyObject>>> {
    let mask_view = mask.as_array();
    let bg_view = bg_median.as_array();
    let height = mask_view.shape()[0];
    let width = mask_view.shape()[1];
    let frame_size = width * height;

    // Need extra frames for temporal stacking [K frames before and K frames after]
    let read_start = if start_frame > k_radius { start_frame - k_radius } else { 0 };
    let read_count = count + 2 * k_radius;
    
    let filter = format!("select='between(n,{},{})'", read_start, read_start + read_count - 1);
    let mut child = Command::new("ffmpeg").args(&["-i", &vid_path, "-vf", &filter, "-f", "rawvideo", "-pix_fmt", "gray", "-vsync", "0", "-"]).stdout(Stdio::piped()).stderr(Stdio::null()).spawn()?;
    let mut stdout = child.stdout.take().unwrap();
    
    let mut all_raw_frames = Vec::new();
    let mut buffer = vec![0u8; frame_size];
    while all_raw_frames.len() < read_count && stdout.read_exact(&mut buffer).is_ok() {
        // Pre-process immediately to f32 to save work in the parallel loop
        let mut processed = vec![0.0f32; frame_size];
        for i in 0..frame_size {
            let pixel = buffer[i] as f32 / 255.0;
            processed[i] = (pixel * mask_view.as_slice().unwrap()[i] - bg_view.as_slice().unwrap()[i]).max(0.0).min(1.0);
        }
        all_raw_frames.push(processed);
    }

    // Now process detections with temporal stacking
    let detections_list: Vec<Vec<[f32; 7]>> = py.allow_threads(|| {
        (0..count)
            .into_par_iter()
            .map(|i| {
                let actual_idx = i + k_radius;
                if actual_idx >= all_raw_frames.len() { return Vec::new(); }
                
                let detections = if k_radius == 0 {
                    steger_detect_frame(&all_raw_frames[actual_idx], width, height, sigma, intensity_threshold, strength_threshold)
                } else {
                    // Perform temporal stacking
                    let low = if actual_idx >= k_radius { actual_idx - k_radius } else { 0 };
                    let high = std::cmp::min(all_raw_frames.len(), actual_idx + k_radius + 1);
                    let num = (high - low) as f32;
                    let mut sum = vec![0.0f32; frame_size];
                    for f_idx in low..high {
                        for p in 0..frame_size { sum[p] += all_raw_frames[f_idx][p]; }
                    }
                    for p in 0..frame_size { sum[p] /= num; }
                    steger_detect_frame(&sum, width, height, sigma, intensity_threshold, strength_threshold)
                };
                detections

            })
            .collect()
    });

    let mut results = Vec::new();
    for dets in detections_list {
        if dets.is_empty() { results.push(None); } else {
            let py_arr = PyArray2::<f32>::zeros(py, [dets.len(), 7], false);
            let mut view = unsafe { py_arr.as_array_mut() };
            for (i, d) in dets.iter().enumerate() { for j in 0..7 { view[[i, j]] = d[j]; } }
            results.push(Some(py_arr.into_any().unbind()));
        }
    }
    Ok(results)
}

#[pyfunction]
#[pyo3(signature = (vid_path, start_frame, count, mask, bg_median))]
fn stream_processed_frames(
    py: Python<'_>,
    vid_path: String,
    start_frame: usize,
    count: usize,
    mask: PyReadonlyArray2<f32>,
    bg_median: PyReadonlyArray2<f32>,
) -> PyResult<PyObject> {
    let mask_view = mask.as_array();
    let bg_view = bg_median.as_array();
    let height = mask_view.shape()[0];
    let width = mask_view.shape()[1];
    let frame_size = width * height;
    let filter = format!("select='between(n,{},{})'", start_frame, start_frame + count - 1);
    let mut child = Command::new("ffmpeg").args(&["-i", &vid_path, "-vf", &filter, "-f", "rawvideo", "-pix_fmt", "gray", "-vsync", "0", "-"]).stdout(Stdio::piped()).stderr(Stdio::null()).spawn()?;
    let mut stdout = child.stdout.take().unwrap();
    let mut buffer = vec![0u8; frame_size];
    let out = PyArray3::<f32>::zeros(py, [count, height, width], false);
    let mut out_view = unsafe { out.as_array_mut() };
    let mut frame_idx = 0;
    while frame_idx < count && py.allow_threads(|| stdout.read_exact(&mut buffer)).is_ok() {
        for r in 0..height {
            for c in 0..width {
                let pixel = buffer[r * width + c] as f32 / 255.0;
                out_view[[frame_idx, r, c]] = (pixel * mask_view[[r, c]] - bg_view[[r, c]]).max(0.0).min(1.0);
            }
        }
        frame_idx += 1;
    }
    Ok(out.into_any().unbind())
}

#[pyfunction]
#[pyo3(signature = (frames_detections, min_frames=2, spatial_radius=2.0, max_gap_frames=1))]
fn temporal_coherence_filter(
    py: Python<'_>,
    frames_detections: Vec<PyReadonlyArray2<f32>>,
    min_frames: usize,
    spatial_radius: f32,
    max_gap_frames: usize,
) -> PyResult<Vec<PyObject>> {
    let num_frames = frames_detections.len();
    if num_frames == 0 { return Ok(vec![]); }
    let spatial_radius_sq = spatial_radius * spatial_radius;
    let trees: Vec<Option<KdTree<f32, 2>>> = frames_detections.iter().map(|dets| {
        let view = dets.as_array();
        if view.shape()[0] == 0 { None } else {
            let mut tree = KdTree::new();
            for (i, row) in view.rows().into_iter().enumerate() { tree.add(&[row[0], row[1]], i as u64); }
            Some(tree)
        }
    }).collect();
    let mut kept: Vec<Vec<bool>> = frames_detections.iter().map(|dets| {
        let view = dets.as_array();
        let n = view.shape()[0];
        let mut k = vec![false; n];
        for i in 0..n { if view[[i, 6]] > 0.5 { k[i] = true; } }
        k
    }).collect();
    for f in 0..num_frames {
        let active_indices: Vec<usize> = kept[f].iter().enumerate().filter(|&(_, &k)| k).map(|(i, _)| i).collect();
        if active_indices.is_empty() { continue; }
        let view_f = frames_detections[f].as_array();
        for k in 1..=max_gap_frames {
            let target_f = f + k;
            if target_f >= num_frames { break; }
            if let Some(ref tree) = trees[target_f] {
                let r_sq = spatial_radius_sq * (k as f32).powi(2);
                for &idx in &active_indices {
                    let pt = [view_f[[idx, 0]], view_f[[idx, 1]]];
                    let neighbors: Vec<NearestNeighbour<f32, u64>> = tree.within::<SquaredEuclidean>(&pt, r_sq);
                    for neighbor in neighbors { kept[target_f][neighbor.item as usize] = true; }
                }
            }
        }
    }
    for f in (0..num_frames).rev() {
        let active_indices: Vec<usize> = kept[f].iter().enumerate().filter(|&(_, &k)| k).map(|(i, _)| i).collect();
        if active_indices.is_empty() { continue; }
        let view_f = frames_detections[f].as_array();
        for k in 1..=max_gap_frames {
            if f < k { break; }
            let target_f = f - k;
            if let Some(ref tree) = trees[target_f] {
                let r_sq = spatial_radius_sq * (k as f32).powi(2);
                for &idx in &active_indices {
                    let pt = [view_f[[idx, 0]], view_f[[idx, 1]]];
                    let neighbors: Vec<NearestNeighbour<f32, u64>> = tree.within::<SquaredEuclidean>(&pt, r_sq);
                    for neighbor in neighbors { kept[target_f][neighbor.item as usize] = true; }
                }
            }
        }
    }
    let mut parent: Vec<usize> = Vec::new();
    let mut node_map: Vec<Vec<Option<usize>>> = Vec::new();
    let mut total_nodes = 0;
    for f in 0..num_frames {
        let n = frames_detections[f].as_array().shape()[0];
        let mut f_map = vec![None; n];
        for i in 0..n {
            if kept[f][i] { f_map[i] = Some(total_nodes); parent.push(total_nodes); total_nodes += 1; }
        }
        node_map.push(f_map);
    }
    fn find(p: &mut Vec<usize>, i: usize) -> usize {
        let mut curr = i;
        while p[curr] != curr { p[curr] = p[p[curr]]; curr = p[curr]; }
        curr
    }
    fn union(p: &mut Vec<usize>, i: usize, j: usize) {
        let root_i = find(p, i); let root_j = find(p, j);
        if root_i != root_j { p[root_i] = root_j; }
    }
    for f in 0..num_frames {
        let view_f = frames_detections[f].as_array();
        for k in 1..=max_gap_frames {
            let nf = f + k; if nf >= num_frames { break; }
            if let Some(ref tree) = trees[nf] {
                let r_sq = spatial_radius_sq * (k as f32).powi(2);
                for i in 0..view_f.shape()[0] {
                    if let Some(u_idx) = node_map[f][i] {
                        let pt = [view_f[[i, 0]], view_f[[i, 1]]];
                        let neighbors: Vec<NearestNeighbour<f32, u64>> = tree.within::<SquaredEuclidean>(&pt, r_sq);
                        for neighbor in neighbors { if let Some(v_idx) = node_map[nf][neighbor.item as usize] { union(&mut parent, u_idx, v_idx); } }
                    }
                }
            }
        }
    }
    let mut component_frames: Vec<HashSet<usize>> = vec![HashSet::new(); total_nodes];
    let mut node_to_comp = vec![0; total_nodes];
    let mut node_counter = 0;
    for f in 0..num_frames {
        for i in 0..frames_detections[f].as_array().shape()[0] {
            if let Some(u_idx) = node_map[f][i] {
                let root = find(&mut parent, u_idx); component_frames[root].insert(f); node_to_comp[node_counter] = root; node_counter += 1;
            }
        }
    }
    let mut results = Vec::new();
    let mut node_counter = 0;
    for f in 0..num_frames {
        let view_f = frames_detections[f].as_array();
        let n = view_f.shape()[0];
        let mut filtered_rows = Vec::new();
        for i in 0..n {
            if let Some(_) = node_map[f][i] {
                let root = node_to_comp[node_counter];
                let persistence = component_frames[root].len();
                if persistence >= min_frames {
                    let mut row = vec![0.0f32; 7]; for c in 0..6 { row[c] = view_f[[i, c]]; } row[6] = persistence as f32; filtered_rows.push(row);
                }
                node_counter += 1;
            }
        }
        let py_array = if filtered_rows.is_empty() { PyArray2::<f32>::zeros(py, [0, 7], false).into_any().unbind() } else {
            let h = filtered_rows.len(); let out = PyArray2::<f32>::zeros(py, [h, 7], false);
            let mut out_view = unsafe { out.as_array_mut() };
            for (r, row) in filtered_rows.iter().enumerate() { for c in 0..7 { out_view[[r, c]] = row[c]; } }
            out.into_any().unbind()
        };
        results.push(py_array);
    }
    Ok(results)
}

#[pyfunction]
#[pyo3(signature = (vid_path, start_frame, end_frame, step=5))]
fn compute_sum_image(py: Python<'_>, vid_path: String, start_frame: usize, end_frame: usize, step: usize) -> PyResult<PyObject> {
    let output = Command::new("ffprobe").args(&["-v", "error", "-select_streams", "v:0", "-show_entries", "stream=width,height", "-of", "csv=p=0:s=x", &vid_path]).output()?;
    let meta = String::from_utf8_lossy(&output.stdout);
    let dims: Vec<&str> = meta.trim().split('x').collect();
    if dims.len() < 2 { return Err(PyErr::new::<pyo3::exceptions::PyRuntimeError, _>("Failed to get video dimensions")); }
    let width: usize = dims[0].parse().unwrap();
    let height: usize = dims[1].parse().unwrap();
    let filter = format!("select='between(n,{},{})*not(mod(n,{}))'", start_frame, end_frame, step);
    let mut child = Command::new("ffmpeg").args(&["-i", &vid_path, "-vf", &filter, "-f", "rawvideo", "-pix_fmt", "gray", "-vsync", "0", "-"]).stdout(Stdio::piped()).stderr(Stdio::null()).spawn()?;
    let mut stdout = child.stdout.take().unwrap();
    let frame_size = width * height;
    let mut buffer = vec![0u8; frame_size];
    let mut accum = vec![0.0f64; frame_size];
    while stdout.read_exact(&mut buffer).is_ok() { for i in 0..frame_size { accum[i] += buffer[i] as f64 / 255.0; } }
    let out = PyArray2::<f64>::zeros(py, [height, width], false);
    let mut out_view = unsafe { out.as_array_mut() };
    for r in 0..height { for c in 0..width { out_view[[r, c]] = accum[r * width + c]; } }
    Ok(out.into_any().unbind())
}

#[pyfunction]
#[pyo3(signature = (vid_path, step=5))]
fn compute_brightness_profile(py: Python<'_>, vid_path: String, step: usize) -> PyResult<(PyObject, PyObject)> {
    let output = Command::new("ffprobe").args(&["-v", "error", "-select_streams", "v:0", "-show_entries", "stream=width,height", "-of", "csv=p=0:s=x", &vid_path]).output()?;
    let meta = String::from_utf8_lossy(&output.stdout);
    let dims: Vec<&str> = meta.trim().split('x').collect();
    let width: usize = dims[0].parse().unwrap();
    let height: usize = dims[1].parse().unwrap();
    let filter = format!("select='not(mod(n,{}))'", step);
    let mut child = Command::new("ffmpeg").args(&["-i", &vid_path, "-vf", &filter, "-f", "rawvideo", "-pix_fmt", "gray", "-vsync", "0", "-"]).stdout(Stdio::piped()).stderr(Stdio::null()).spawn()?;
    let mut stdout = child.stdout.take().unwrap();
    let frame_size = width * height;
    let mut buffer = vec![0u8; frame_size];
    let mut brightness = Vec::new(); let mut frames = Vec::new();
    let mut current_frame = 0;
    while stdout.read_exact(&mut buffer).is_ok() {
        let sum: f64 = buffer.iter().map(|&b| b as f64 / 255.0).sum();
        brightness.push(sum); frames.push(current_frame as f32); current_frame += step;
    }
    let b_out = numpy::PyArray1::<f64>::from_vec(py, brightness).into_any().unbind();
    let f_out = numpy::PyArray1::<f32>::from_vec(py, frames).into_any().unbind();
    Ok((f_out, b_out))
}

#[pymodule]
fn rust_core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(temporal_coherence_filter, m)?)?;
    m.add_function(wrap_pyfunction!(compute_sum_image, m)?)?;
    m.add_function(wrap_pyfunction!(compute_brightness_profile, m)?)?;
    m.add_function(wrap_pyfunction!(process_video_steger, m)?)?;
    m.add_function(wrap_pyfunction!(stream_processed_frames, m)?)?;
    Ok(())
}
