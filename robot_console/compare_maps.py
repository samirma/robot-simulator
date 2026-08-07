import cv2
import numpy as np

def load_map(yaml_path):
    info = {}
    with open(yaml_path, 'r') as f:
        for line in f:
            if ':' in line:
                key, val = line.strip().split(':', 1)
                key = key.strip()
                val = val.strip()
                if key == 'origin':
                    val = val.strip('[]').split(',')
                    info[key] = [float(v.strip()) for v in val]
                elif key == 'resolution':
                    info[key] = float(val)
                elif key in ('occupied_thresh', 'free_thresh'):
                    info[key] = float(val)
                elif key == 'negate':
                    info[key] = int(val)
                else:
                    info[key] = val
    img_path = yaml_path.replace('.yaml', '.pgm')
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    return img, info

img_gt, info_gt = load_map('runs/house_gt/map.yaml')
img_slam, info_slam = load_map('runs/house_vslam/map.yaml')

res = info_gt['resolution']

# origins
orig_gt_x, orig_gt_y = info_gt['origin'][:2]
orig_slam_x, orig_slam_y = info_slam['origin'][:2]

# Map coordinate in world:
# x_world = orig_x + x_px * res
# y_world = orig_y + (height - 1 - y_px) * res (Assuming standard map_server: origin is bottom-left pixel)
# Actually, map_server origin is the bottom-left corner of the pixel at (0, img.shape[0]-1). Let's just align them based on origin.
h_gt, w_gt = img_gt.shape
h_slam, w_slam = img_slam.shape

# find bounds to merge
min_x = min(orig_gt_x, orig_slam_x)
min_y = min(orig_gt_y, orig_slam_y)
max_x = max(orig_gt_x + w_gt * res, orig_slam_x + w_slam * res)
max_y = max(orig_gt_y + h_gt * res, orig_slam_y + h_slam * res)

w_merged = int(round((max_x - min_x) / res))
h_merged = int(round((max_y - min_y) / res))

def place_in_merged(img, info, min_x, min_y, w_m, h_m):
    orig_x, orig_y = info['origin'][:2]
    h, w = img.shape
    
    # x offset in pixels
    dx = int(round((orig_x - min_x) / res))
    # y offset from bottom
    dy_bottom = int(round((orig_y - min_y) / res))
    
    # in image coordinates, y=0 is top. 
    # bottom is h_m - 1
    out = np.full((h_m, w_m), 205, dtype=np.uint8) # 205 is unknown usually
    
    y_start = h_m - dy_bottom - h
    y_end = h_m - dy_bottom
    x_start = dx
    x_end = dx + w
    
    out[y_start:y_end, x_start:x_end] = img
    return out

gt_aligned = place_in_merged(img_gt, info_gt, min_x, min_y, w_merged, h_merged)
slam_aligned = place_in_merged(img_slam, info_slam, min_x, min_y, w_merged, h_merged)

# Calculate difference
# states: free (254), occupied (0), unknown (205)
# let's just do a pixel-wise difference on known regions of GT
# known in GT
mask_gt_known = gt_aligned != 205
mask_slam_known = slam_aligned != 205

# intersection of known
mask_both = mask_gt_known & mask_slam_known

# difference in both known
diff = np.abs(gt_aligned.astype(np.int32) - slam_aligned.astype(np.int32))
mse = np.mean(diff[mask_both] ** 2)
mae = np.mean(diff[mask_both])

print(f"GT size: {w_gt}x{h_gt}, SLAM size: {w_slam}x{h_slam}")
print(f"Aligned size: {w_merged}x{h_merged}")
print(f"Known pixels in GT: {np.sum(mask_gt_known)}")
print(f"Known pixels in SLAM: {np.sum(mask_slam_known)}")
print(f"Overlap known pixels: {np.sum(mask_both)}")
print(f"Mean Absolute Error (0-255) in overlap: {mae:.2f}")
print(f"MSE in overlap: {mse:.2f}")

# Threshold diffs
diff_thresh = diff[mask_both] > 50
print(f"Pixels with significant difference (>50): {np.sum(diff_thresh)} ({np.sum(diff_thresh)/np.sum(mask_both)*100:.2f}%)")

# Compare occupied pixels
gt_occ = gt_aligned < 50
slam_occ = slam_aligned < 50

# IOU of occupied pixels
intersection_occ = gt_occ & slam_occ
union_occ = gt_occ | slam_occ
iou_occ = np.sum(intersection_occ) / np.sum(union_occ) if np.sum(union_occ) > 0 else 0
print(f"Intersection Over Union of occupied pixels: {iou_occ:.4f}")

# Visualizing the difference
out_color = np.zeros((h_merged, w_merged, 3), dtype=np.uint8)
out_color[gt_occ] = [0, 255, 0] # GT is green
out_color[slam_occ] = [0, 0, 255] # SLAM is red
out_color[intersection_occ] = [255, 255, 255] # Both is white

cv2.imwrite('runs/compare_occ_vslam.png', out_color)
print("Saved comparison image to runs/compare_occ_vslam.png")
