import cv2
import numpy as np
import torch
from ultralytics import YOLO

# --------------------------
# Config
# --------------------------
FRONT_VIDEO = r"C:\Users\azmat\f_new.mp4"
BACK_VIDEO  = r"C:\Users\azmat\b_new.mp4"

YOLO_WEIGHTS = "yolov8n.pt"
CONF_THRESH = 0.35
DISPLAY_SCALE = 0.3

# Matching thresholds
REID_SIM_THRESHOLD = 0.75
HIST_SIM_THRESHOLD = 0.88
GLOBAL_REID_THRESHOLD = 0.72
GLOBAL_HIST_THRESHOLD = 0.86

# Movement thresholds (pixels/frame)
MOVE_THRESHOLD = 30

# Appearance update momentum
APPEARANCE_MOMENTUM = 0.6

# Suspicion hyper-parameters
SUSPICION_SCORE_THRESHOLD = 3     # final score threshold to mark suspicious
ANGLE_BIG = 45
ANGLE_SMALL = 25
SIZE_CHANGE_RATIO = 0.35          # > 35% change flagged as sudden
UNSTABLE_HISTORY_THRESH = 3       # number of low-sim events to consider unstable
UNSTABLE_SIM_CUTOFF = 0.50       # similarity below this counts as instability

# --------------------------
# Try to load TorchReID (preferred)
# --------------------------
REID_AVAILABLE = False
reid_model = None
reid_transform = None
device = "cuda" if torch.cuda.is_available() else "cpu"

try:
    import torchreid
    REID_AVAILABLE = True

    reid_model = torchreid.utils.tools.build_model(
        name='osnet_x1_0', num_classes=1000, pretrained=True
    )
    reid_model.eval().to(device)

    from torchvision import transforms
    reid_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((256, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    print("[INFO] TorchReID loaded: using OSNet embeddings for appearance matching.")
except Exception as e:
    print(f"[INFO] TorchReID not available ({e}); falling back to color histograms.")

# --------------------------
# Utilities
# --------------------------
def extract_histogram(bgr_crop, bins=(8, 8, 8)):
    if bgr_crop is None or bgr_crop.size == 0:
        return np.zeros(bins[0]*bins[1]*bins[2], dtype=np.float32)
    hsv = cv2.cvtColor(bgr_crop, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1, 2], None, bins, [0, 180, 0, 256, 0, 256])
    cv2.normalize(hist, hist)
    return hist.flatten().astype(np.float32)

def cosine_similarity(vec1, vec2):
    a = np.asarray(vec1).astype(np.float32)
    b = np.asarray(vec2).astype(np.float32)
    denom = (np.linalg.norm(a) * np.linalg.norm(b)) + 1e-8
    return float(np.dot(a, b) / denom)

@torch.no_grad()
def extract_reid_embedding(bgr_crop):
    if reid_model is None or reid_transform is None:
        return None
    try:
        img = reid_transform(bgr_crop)
    except Exception:
        from torchvision import transforms
        fallback = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((256, 128)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
        img = fallback(bgr_crop)
    img = img.unsqueeze(0).to(device)
    feat = reid_model(img)
    if isinstance(feat, (list, tuple)):
        feat = feat[0]
    feat = feat.squeeze().detach().cpu().numpy().astype(np.float32)
    norm = np.linalg.norm(feat) + 1e-8
    return feat / norm

def compute_centroid(box):
    x1, y1, x2, y2 = box
    return ((x1 + x2) // 2, (y1 + y2) // 2)

def movement_score(prev_box, curr_box):
    if prev_box is None or curr_box is None:
        return 0.0
    c1 = compute_centroid(prev_box)
    c2 = compute_centroid(curr_box)
    dx = c2[0] - c1[0]
    dy = c2[1] - c1[1]
    return float(np.sqrt(dx*dx + dy*dy))

def bbox_area(box):
    if box is None:
        return 0.0
    x1, y1, x2, y2 = box
    return float(max(0, x2 - x1) * max(0, y2 - y1))

# crude orientation estimator based on aspect ratio (replaceable by pose)
def estimate_orientation(box):
    if box is None:
        return 0.0
    x1, y1, x2, y2 = box
    w = max(1.0, float(x2 - x1))
    h = max(1.0, float(y2 - y1))
    aspect = h / w
    # heuristic: taller -> facing camera (front), shorter/wider -> back or side
    if aspect > 2.8:
        return 0.0   # front
    if aspect < 2.0:
        return 180.0 # back
    return 90.0     # sideways

# --------------------------
# Detection
# --------------------------
yolo_model = YOLO(YOLO_WEIGHTS)

def detect_people(frame):
    results = yolo_model.predict(source=frame, imgsz=640, conf=CONF_THRESH, verbose=False)
    dets = []
    h, w = frame.shape[:2]

    for r in results:
        names = r.names
        for b in r.boxes:
            cls_id = int(b.cls.item())
            label = names[cls_id]
            if label != "person":
                continue
            x1, y1, x2, y2 = map(int, b.xyxy[0].tolist())
            x1 = max(0, min(x1, w - 1)); x2 = max(0, min(x2, w - 1))
            y1 = max(0, min(y1, h - 1)); y2 = max(0, min(y2, h - 1))
            if x2 <= x1 or y2 <= y1:
                continue

            crop = frame[y1:y2, x1:x2].copy()
            emb = extract_reid_embedding(crop) if REID_AVAILABLE else None
            hist = extract_histogram(crop)

            dets.append({
                "box": (x1, y1, x2, y2),
                "emb": emb,
                "hist": hist
            })

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 220, 0), 2)
            cv2.putText(frame, "person", (x1, max(0, y1 - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 220, 0), 2)
    return frame, dets

# --------------------------
# Matching & Fusion (per-camera)
# --------------------------
def match_across_cameras(front_dets, back_dets):
    fused = []
    used_back = set()

    for i, f in enumerate(front_dets):
        best_j = -1
        best_score = -1.0
        best_threshold = None
        best_method = "Histogram"
        for j, b in enumerate(back_dets):
            if j in used_back:
                continue

            if f["emb"] is not None and b["emb"] is not None:
                score = cosine_similarity(f["emb"], b["emb"])
                threshold = REID_SIM_THRESHOLD
                method = "ReID"
            else:
                score = cosine_similarity(f["hist"], b["hist"])
                threshold = HIST_SIM_THRESHOLD
                method = "Histogram"

            if score > best_score:
                best_score = score
                best_j = j
                best_threshold = threshold
                best_method = method

        if best_j != -1 and best_score >= best_threshold:
            used_back.add(best_j)
            fused.append({
                "front_idx": i,
                "back_idx": best_j,
                "similarity": float(round(best_score, 3)),
                "method": best_method
            })
    return fused

# --------------------------
# Global ID assignment
# --------------------------
global_id_counter = 0
global_tracks = {}  # gid -> {"vec":..., "type":"emb"|"hist", "last_seen":frame_idx, "vis_count":int}

def assign_global_ids(front_dets, back_dets, fused_matches, frame_idx):
    global global_id_counter, global_tracks

    matches_with_global = []
    used_front = set()
    used_back = set()

    def find_best_global(vec, vec_type):
        best_gid = None
        best_score = -1.0
        for gid, meta in global_tracks.items():
            if meta["type"] != vec_type:
                continue
            score = cosine_similarity(vec, meta["vec"])
            if score > best_score:
                best_score = score
                best_gid = gid
        return best_gid, best_score

    # fused pairs
    for m in fused_matches:
        fi = m["front_idx"]
        bi = m["back_idx"]
        used_front.add(fi)
        used_back.add(bi)

        f_vec = front_dets[fi]["emb"] if front_dets[fi]["emb"] is not None else front_dets[fi]["hist"]
        vec_type = "emb" if front_dets[fi]["emb"] is not None else "hist"

        gid, score = find_best_global(f_vec, vec_type)
        threshold = GLOBAL_REID_THRESHOLD if vec_type == "emb" else GLOBAL_HIST_THRESHOLD

        if gid is not None and score >= threshold:
            assigned_gid = gid
            old = global_tracks[assigned_gid]["vec"]
            new_vec = APPEARANCE_MOMENTUM * old + (1.0 - APPEARANCE_MOMENTUM) * f_vec
            if vec_type == "emb":
                norm = np.linalg.norm(new_vec) + 1e-8
                new_vec = new_vec / norm
            global_tracks[assigned_gid]["vec"] = new_vec
            global_tracks[assigned_gid]["last_seen"] = frame_idx
            global_tracks[assigned_gid]["vis_count"] += 1
        else:
            global_id_counter += 1
            assigned_gid = global_id_counter
            vec_to_store = f_vec.copy()
            if vec_type == "emb":
                n = np.linalg.norm(vec_to_store) + 1e-8
                vec_to_store = vec_to_store / n
            global_tracks[assigned_gid] = {
                "vec": vec_to_store,
                "type": vec_type,
                "last_seen": frame_idx,
                "vis_count": 1
            }

        matches_with_global.append({
            "global_id": assigned_gid,
            "front_idx": fi,
            "back_idx": bi,
            "similarity": m.get("similarity", None),
            "method": m.get("method", vec_type)
        })

    # unmatched front-only
    for i, f in enumerate(front_dets):
        if i in used_front:
            continue
        f_vec = f["emb"] if f["emb"] is not None else f["hist"]
        vec_type = "emb" if f["emb"] is not None else "hist"
        gid, score = find_best_global(f_vec, vec_type)
        threshold = GLOBAL_REID_THRESHOLD if vec_type == "emb" else GLOBAL_HIST_THRESHOLD
        if gid is not None and score >= threshold:
            assigned_gid = gid
            old = global_tracks[assigned_gid]["vec"]
            new_vec = APPEARANCE_MOMENTUM * old + (1.0 - APPEARANCE_MOMENTUM) * f_vec
            if vec_type == "emb":
                norm = np.linalg.norm(new_vec) + 1e-8
                new_vec = new_vec / norm
            global_tracks[assigned_gid]["vec"] = new_vec
            global_tracks[assigned_gid]["last_seen"] = frame_idx
            global_tracks[assigned_gid]["vis_count"] += 1
        else:
            global_id_counter += 1
            assigned_gid = global_id_counter
            vec_to_store = f_vec.copy()
            if vec_type == "emb":
                n = np.linalg.norm(vec_to_store) + 1e-8
                vec_to_store = vec_to_store / n
            global_tracks[assigned_gid] = {
                "vec": vec_to_store,
                "type": vec_type,
                "last_seen": frame_idx,
                "vis_count": 1
            }
        matches_with_global.append({
            "global_id": assigned_gid,
            "front_idx": i,
            "back_idx": None,
            "similarity": None,
            "method": vec_type
        })

    # unmatched back-only
    for j, b in enumerate(back_dets):
        if j in used_back:
            continue
        b_vec = b["emb"] if b["emb"] is not None else b["hist"]
        vec_type = "emb" if b["emb"] is not None else "hist"
        gid, score = find_best_global(b_vec, vec_type)
        threshold = GLOBAL_REID_THRESHOLD if vec_type == "emb" else GLOBAL_HIST_THRESHOLD
        if gid is not None and score >= threshold:
            assigned_gid = gid
            old = global_tracks[assigned_gid]["vec"]
            new_vec = APPEARANCE_MOMENTUM * old + (1.0 - APPEARANCE_MOMENTUM) * b_vec
            if vec_type == "emb":
                norm = np.linalg.norm(new_vec) + 1e-8
                new_vec = new_vec / norm
            global_tracks[assigned_gid]["vec"] = new_vec
            global_tracks[assigned_gid]["last_seen"] = frame_idx
            global_tracks[assigned_gid]["vis_count"] += 1
        else:
            global_id_counter += 1
            assigned_gid = global_id_counter
            vec_to_store = b_vec.copy()
            if vec_type == "emb":
                n = np.linalg.norm(vec_to_store) + 1e-8
                vec_to_store = vec_to_store / n
            global_tracks[assigned_gid] = {
                "vec": vec_to_store,
                "type": vec_type,
                "last_seen": frame_idx,
                "vis_count": 1
            }
        matches_with_global.append({
            "global_id": assigned_gid,
            "front_idx": None,
            "back_idx": j,
            "similarity": None,
            "method": vec_type
        })

    return matches_with_global

def draw_global_ids(frame, dets, prefix="F", matches=None, which_camera="front"):
    idx2gid = {}
    if matches is not None:
        for m in matches:
            if which_camera == "front" and m["front_idx"] is not None:
                idx2gid[m["front_idx"]] = m["global_id"]
            if which_camera == "back" and m["back_idx"] is not None:
                idx2gid[m["back_idx"]] = m["global_id"]

    for idx, d in enumerate(dets):
        x1, y1, x2, y2 = d["box"]
        gid = idx2gid.get(idx, None)
        label = f"{prefix}{idx}" if gid is None else f"ID{gid}"
        cv2.putText(frame, label, (x1, y2 + 16),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 180, 0), 2)

# --------------------------
# Enhanced Suspicion Logic
# --------------------------
# maintain small histories
prev_positions = {}        # gid -> last known box (front-preferred)
orientation_history = {}   # gid -> last orientation estimate
size_history = {}          # gid -> last bbox area
instability_counter = {}   # gid -> counter of low-sim events

def is_suspicious_enhanced(movement, similarity, method, angle_change, unstable_id, sudden_size_change):
    score = 0

    # Movement
    if movement > (MOVE_THRESHOLD * 2):
        score += 2
    elif movement > MOVE_THRESHOLD:
        score += 1

    # Appearance similarity
    if similarity is not None:
        if similarity < 0.55:
            score += 2
        elif similarity < 0.70:
            score += 1

    # Angle change (turning)
    if angle_change > ANGLE_BIG:
        score += 2
    elif angle_change > ANGLE_SMALL:
        score += 1

    # Unstable ID history
    if unstable_id:
        score += 2

    # Sudden size change
    if sudden_size_change:
        score += 1

    return score >= SUSPICION_SCORE_THRESHOLD

# --------------------------
# Main
# --------------------------
def main():
    front_cap = cv2.VideoCapture(FRONT_VIDEO)
    back_cap  = cv2.VideoCapture(BACK_VIDEO)

    if not front_cap.isOpened():
        raise ValueError("Front video not opened properly")
    if not back_cap.isOpened():
        raise ValueError("Back video not opened properly")

    print("[INFO] Dual-video ReID fusion started. Press ESC to quit.")

    frame_idx = 0

    while True:
        ret_f, frame_f = front_cap.read()
        ret_b, frame_b = back_cap.read()
        if not ret_f or not ret_b:
            break

        frame_f, dets_f = detect_people(frame_f)
        frame_b, dets_b = detect_people(frame_b)

        # Optional: draw temporary per-camera ids
        draw_global_ids(frame_f, dets_f, prefix="F", matches=None, which_camera="front")
        draw_global_ids(frame_b, dets_b, prefix="B", matches=None, which_camera="back")

        fused = match_across_cameras(dets_f, dets_b)

        matches = assign_global_ids(dets_f, dets_b, fused, frame_idx)

        # redraw using global IDs
        draw_global_ids(frame_f, dets_f, prefix="F", matches=matches, which_camera="front")
        draw_global_ids(frame_b, dets_b, prefix="B", matches=matches, which_camera="back")

        # analyze matches and decide suspiciousness
        for m in matches:
            gid = m["global_id"]
            fx_box = None
            bx_box = None
            if m["front_idx"] is not None:
                fx_box = dets_f[m["front_idx"]]["box"]
            if m["back_idx"] is not None:
                bx_box = dets_b[m["back_idx"]]["box"]

            # choose box for movement tracking (prefer front)
            curr_box = fx_box if fx_box is not None else bx_box
            prev_box = prev_positions.get(gid, curr_box)
            move = movement_score(prev_box, curr_box)
            prev_positions[gid] = curr_box

            # similarity & method
            similarity = m.get("similarity", None)
            method = m.get("method", "ReID")

            # orientation
            curr_orientation = estimate_orientation(curr_box)
            prev_orientation = orientation_history.get(gid, curr_orientation)
            angle_change = abs(curr_orientation - prev_orientation)
            orientation_history[gid] = curr_orientation

            # size change
            prev_area = size_history.get(gid, bbox_area(prev_box))
            curr_area = bbox_area(curr_box)
            size_history[gid] = curr_area
            size_change_ratio = abs(curr_area - prev_area) / (prev_area + 1e-8)
            sudden_size_change = size_change_ratio > SIZE_CHANGE_RATIO

            # instability tracking
            if similarity is not None and similarity < UNSTABLE_SIM_CUTOFF:
                instability_counter[gid] = instability_counter.get(gid, 0) + 1
            else:
                instability_counter[gid] = max(0, instability_counter.get(gid, 0) - 1)
            unstable_id = instability_counter[gid] >= UNSTABLE_HISTORY_THRESH

            suspicious = is_suspicious_enhanced(
                movement=move,
                similarity=similarity,
                method=method,
                angle_change=angle_change,
                unstable_id=unstable_id,
                sudden_size_change=sudden_size_change
            )

            # Logging
            print(
                f"[MATCH] GID={gid} | sim={similarity} | method={method} "
                f"| front={fx_box} back={bx_box} | move={round(move, 2)} | angle_ch={angle_change} "
                f"| size_ch={round(size_change_ratio, 2)} | unstable_cnt={instability_counter.get(gid,0)} | suspicious={suspicious}"
            )

            # Draw annotations
            if fx_box is not None:
                fx1, fy1, fx2, fy2 = fx_box
                if suspicious:
                    cv2.putText(frame_f,
                                f"SUSPICIOUS ID{gid} ({similarity}, {method})",
                                (fx1, max(0, fy1 - 40)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    cv2.rectangle(frame_f, (fx1, fy1), (fx2, fy2), (0, 0, 255), 2)
                else:
                    cv2.putText(frame_f,
                                f"ID{gid} ({similarity}, {method})",
                                (fx1, max(0, fy1 - 24)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 140, 255), 2)

            if bx_box is not None:
                bx1, by1, bx2, by2 = bx_box
                if suspicious:
                    cv2.putText(frame_b,
                                f"SUSPICIOUS ID{gid} ({similarity}, {method})",
                                (bx1, max(0, by1 - 40)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    cv2.rectangle(frame_b, (bx1, by1), (bx2, by2), (0, 0, 255), 2)
                else:
                    cv2.putText(frame_b,
                                f"ID{gid} ({similarity}, {method})",
                                (bx1, max(0, by1 - 24)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 140, 255), 2)

        # Side-by-side display
        h = min(frame_f.shape[0], frame_b.shape[0])
        frame_f_disp = cv2.resize(frame_f, (int(frame_f.shape[1]*h/frame_f.shape[0]), h))
        frame_b_disp = cv2.resize(frame_b, (int(frame_b.shape[1]*h/frame_b.shape[0]), h))
        combined = cv2.hconcat([frame_f_disp, frame_b_disp])

        new_w = int(combined.shape[1] * DISPLAY_SCALE)
        new_h = int(combined.shape[0] * DISPLAY_SCALE)
        combined = cv2.resize(combined, (new_w, new_h))

        cv2.imshow("Front | Back (Global-ID ReID Fusion + Suspicion)", combined)
        if cv2.waitKey(1) & 0xFF == 27:
            break

        frame_idx += 1

    front_cap.release()
    back_cap.release()
    cv2.destroyAllWindows()
    print("[INFO] Finished.")

if __name__ == "__main__":
    main()
