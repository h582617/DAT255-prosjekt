import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import math
import os

img_size = 416
grid_size = 13
num_classes = 4 
batch_size = 32
epochs = 40
anchors = np.array([[30,30], [40,40], [60, 60]], dtype=np.float32)
anchor_count = anchors.shape[0]
iou_threshold = 0.5
score_threshold = 0.05
nms_iou_threshold = 0.45

def xywh_to_corners_np(box):
    cx, cy, w, h = box
    x1 = cx - w/2.0
    y1 = cy - h/2.0
    x2 = cx + w/2.0
    y2 = cy + h/2.0
    return np.array([x1, y1, x2, y2])

def iou_np(box1, box2):
    xa1 = max(box1[0], box2[0])
    ya1 = max(box1[1], box2[1])
    xa2 = min(box1[2], box2[2])
    ya2 = min(box1[3], box2[3])
    inter_w = max(xa2 - xa1, 0.)
    inter_h = max(ya2 - ya1, 0.)
    inter = inter_w * inter_h
    a1 = max(0., box1[2]-box1[0]) * max(0., box1[3]-box1[1])
    a2 = max(0., box2[2]-box2[0]) * max(0., box2[3]-box2[1])
    union = a1 + a2 - inter + 1e-9
    return inter/union if union > 0 else 0.0

def create_yolo_model(input_shape=(img_size, img_size, 3), num_classes=num_classes, anchors=anchors): 
    
    inputs = layers.Input(shape=input_shape)
    x = layers.Conv2D(32, 3, padding="same", activation="relu")(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D(2)(x)

    x = layers.Conv2D(64, 3, padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D(2)(x)

    x = layers.Conv2D(128, 3, padding="same", activation="relu" )(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D(2)(x)
    
    x = layers.Conv2D(256, 3, padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D(2)(x)
    
    x = layers.Conv2D(512, 3, padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(512, 3, strides=2, padding="same", activation="relu")(x)

    output_filters = anchor_count * (5 + num_classes)
    outputs = layers.Conv2D(output_filters, kernel_size=1, padding="same")(x)

    model = models.Model(inputs, outputs)
    return model

def encode_labels(boxes, classes, img_size=img_size, grid_size=grid_size, anchors=anchors, num_classes=num_classes):
    
    anchor_count = anchors.shape[0]
    y = np.zeros((grid_size, grid_size, anchor_count, 5 + num_classes), dtype=np.float32)
    cell_size = img_size / grid_size

    for (cx, cy, w, h), cls in zip(boxes, classes):
        gx = int(cx // cell_size)
        gy = int(cy // cell_size)
        if gx < 0 or gx >= grid_size or gy < 0 or gy >= grid_size:
            continue

        best_iou = -1
        best_idx = 0
        bbox_corners = xywh_to_corners_np([cx,cy,w,h])
        for i,a in enumerate(anchors):
            aw,ah = a
            anchor_corners = xywh_to_corners_np([cx,cy,aw,ah])
            iou = iou_np(bbox_corners, anchor_corners)
            if iou > best_iou:
                best_iou = iou
                best_idx = i 
        
        tx = (cx / cell_size) - gx
        ty = (cy / cell_size) - gy
        tw = math.log((w + 1e-9) / (anchors[best_idx][0] + 1e-9))
        th = math.log((h + 1e-9) / (anchors[best_idx][1] + 1e-9))

        y[gy, gx, best_idx, 0:4] = [tx, ty, tw, th]
        y[gy, gx, best_idx, 4] = 1.0
        y[gy, gx, best_idx, 5 + cls] = 1.0
    return y

def yolo_loss(anchors=anchors, grid_size=grid_size, num_classes=num_classes):
    anchor_count = anchors.shape[0]
    cce = tf.keras.losses.CategoricalCrossentropy(from_logits=True, reduction=tf.keras.losses.Reduction.NONE) 

    def loss(y_true, y_pred):
        batch = tf.shape(y_pred)[0]
        feat = tf.reshape(y_pred, (batch, grid_size, grid_size, anchor_count, 5 + num_classes))

        pred_box = feat[..., 0:4]
        pred_obj = feat[..., 4:5]
        pred_class = feat[..., 5:]

        true_box = y_true[..., 0:4]
        true_obj = y_true[..., 4:5]
        true_class = y_true[..., 5:]

        true_obj_s = tf.squeeze(true_obj, axis=-1)
        pred_obj_s = tf.squeeze(pred_obj, axis=-1)

        obj_loss_raw = tf.nn.sigmoid_cross_entropy_with_logits(labels=true_obj_s, logits=pred_obj_s)
        positive_mask = tf.equal(true_obj_s, 1.0)
        obj_loss_weighted = tf.where(positive_mask, obj_loss_raw, 0.5 * obj_loss_raw)
        obj_loss = tf.reduce_sum(obj_loss_weighted)

        box_loss = tf.reduce_sum(true_obj * tf.square(true_box - pred_box))

        class_loss_raw = cce(true_class, pred_class)
        class_loss_masked = class_loss_raw * true_obj_s
        class_loss = tf.reduce_sum(class_loss_masked)

        total = box_loss + obj_loss + class_loss
        total = total / tf.cast(batch, tf.float32)
        return total
    return loss

def augment(image):
    image = tf.image.convert_image_dtype(image, tf.float32)
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_brightness(image, 0.1)
    image = tf.clip_by_value(image, 0.0, 1.0)
    return image

def create_dataset(image_paths, label_paths, batch_size=batch_size, training=True, shuffle_buffer=1000):
    ds = tf.data.Dataset.from_tensor_slices((image_paths, label_paths))

    def _load(image_path, label_path):
        img = tf.io.read_file(image_path)
        img = tf.image.decode_jpeg(img, channels=3)
        img = tf.image.resize(img, (img_size, img_size))
        img = tf.cast(img, tf.float32) / 255.0
        label = tf.py_function(lambda p: np.load(p.numpy().decode('utf-8')).astype(np.float32),
                               [label_path], Tout=tf.float32)
        label.set_shape((grid_size, grid_size, anchor_count, 5 + num_classes))
        return img, label

    ds = ds.map(_load, num_parallel_calls=tf.data.AUTOTUNE)
    if training:
        ds = ds.shuffle(shuffle_buffer)
        ds = ds.map(lambda x, y: (augment(x), y), num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds

def softmax(x, axis=-1):
    x = np.asarray(x)
    x = x - np.max(x, axis=axis, keepdims=True)
    e = np.exp(x)
    return e / (np.sum(e, axis=axis, keepdims=True) + 1e-9)

def sigmoid(x):
    x = np.asarray(x)
    return 1.0 / (1.0 + np.exp(-x))

def decode_predictions(pred, img_size=img_size, grid_size=grid_size, anchors=anchors, num_classes=num_classes): 
    if pred.ndim == 4: 
        pred = np.squeeze(pred, axis=0)
    pred = pred.reshape((grid_size, grid_size, anchor_count, 5 + num_classes))
    cell_size = img_size / grid_size
    boxes = []
    for gy in range(grid_size):
        for gx in range(grid_size):
            for a in range(anchor_count):
                tx,ty,tw,th = pred[gy,gx,a,0:4]
                obj_logit = pred[gy,gx,a,4]
                class_logits = pred[gy,gx,a,5:]
                score = sigmoid(obj_logit)
                if score < score_threshold:
                    continue
                cx = (gx + sigmoid(tx)) * cell_size
                cy = (gy + sigmoid(ty)) * cell_size
                w = math.exp(tw) * anchors[a][0]
                h = math.exp(th) * anchors[a][1]
                x1 = cx - w/2.0
                y1 = cy - h/2.0
                x2 = cx + w/2.0
                y2 = cy + h/2.0
                class_id = int(np.argmax(class_logits))
                class_score = softmax(class_logits)[class_id]
                final_score = score * class_score
                boxes.append([x1,y1,x2,y2, final_score, class_id])
   
    final = []
    for c in range(num_classes):
        cls_boxes = [b for b in boxes if b[5] == c]
        if not cls_boxes:
            continue
        cls_boxes.sort(key=lambda x: x[4], reverse=True)
        kept = []
        while cls_boxes:
            box = cls_boxes.pop(0)
            kept.append(box)
            cls_boxes = [b for b in cls_boxes if iou_np(box[:4], b[:4]) < nms_iou_threshold]
        final.extend(kept)
    return final

def voc_ap_from_preds(gt_by_image, pred_by_image, num_classes, iou_thresh=0.5):
    gt_count_per_class = {c: 0 for c in range(num_classes)}
    gt_per_class = {c: {} for c in range(num_classes)}
    for img_idx, gts in enumerate(gt_by_image):
        for box in gts:
            x1,y1,x2,y2,cid = box
            gt_count_per_class[cid] += 1
            gt_per_class[cid].setdefault(img_idx, []).append([x1,y1,x2,y2, False])
    APs = {}
    PR_curves = {}
    for c in range(num_classes):
        preds = []
        for img_idx, preds_img in enumerate(pred_by_image):
            for p in preds_img:
                x1,y1,x2,y2,score,cid = p
                if cid != c: continue
                preds.append((img_idx, score, [x1,y1,x2,y2]))
        if len(preds) == 0:
            APs[c] = 0.0
            PR_curves[c] = {"precision": np.array([0.0]), "recall": np.array([0.0])}
            continue
        preds.sort(key=lambda x: x[1], reverse=True)
        tp = np.zeros(len(preds))
        fp = np.zeros(len(preds))
        for i, (img_idx, score, box) in enumerate(preds):
            gts = gt_per_class[c].get(img_idx, [])
            best_iou = 0.0
            best_j = -1
            for j, gt in enumerate(gts):
                iouv = iou_xyxy(box, gt[:4])
                if iouv > best_iou:
                    best_iou = iouv
                    best_j = j
            if best_iou >= iou_thresh and best_j >= 0 and not gts[best_j][4]:
                tp[i] = 1
                gts[best_j][4] = True
            else:
                fp[i] = 1
        fp_cum = np.cumsum(fp)
        tp_cum = np.cumsum(tp)
        if gt_count_per_class[c] == 0:
            recall = tp_cum * 0.0
        else:
            recall = tp_cum / float(gt_count_per_class[c])
        precision = tp_cum / (tp_cum + fp_cum + 1e-9)
        AP = compute_ap(recall, precision)
        APs[c] = AP
        PR_curves[c] = {"precision": precision, "recall": recall}
    mAP = np.mean(list(APs.values())) if APs else 0.0
    return {"AP_per_class": APs, "mAP": mAP, "PR_curves": PR_curves}

def precision_recall_at_confidence(gt_by_image, pred_by_image, num_classes, conf_thresh=0.5, iou_thresh=0.5):
    
    gt_per_class = {c: {} for c in range(num_classes)}
    gt_count_per_class = {c: 0 for c in range(num_classes)}
    for img_idx, gts in enumerate(gt_by_image):
        for g in gts:
            x1,y1,x2,y2,cid = g
            gt_count_per_class[cid] += 1
            gt_per_class[cid].setdefault(img_idx, []).append([x1,y1,x2,y2, False])

    results = {}
    for c in range(num_classes):
        tp = 0
        fp = 0
        for img_idx, preds in enumerate(pred_by_image):
            preds_c = [p for p in preds if p[5] == c and p[4] >= conf_thresh]
            preds_c.sort(key=lambda x: x[4], reverse=True)
            gts_img = gt_per_class[c].get(img_idx, [])
            for p in preds_c:
                box = p[0:4]
                best_iou = 0.0
                best_j = -1
                for j, gt in enumerate(gts_img):
                    iouv = iou_xyxy(box, gt[:4])
                    if iouv > best_iou:
                        best_iou = iouv
                        best_j = j
                if best_iou >= iou_thresh and best_j >= 0 and not gts_img[best_j][4]:
                    tp += 1
                    gts_img[best_j][4] = True
                else:
                    fp += 1
        fn = gt_count_per_class[c] - tp
        precision = tp / (tp + fp + 1e-9) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn + 1e-9) if (tp + fn) > 0 else 0.0
        results[c] = {"precision": float(precision), "recall": float(recall), "tp": int(tp), "fp": int(fp), "fn": int(max(0, fn))}
    return results

def iou_xyxy(a, b):
    xa1, ya1, xa2, ya2 = a
    xb1, yb1, xb2, yb2 = b
    xi1 = max(xa1, xb1)
    yi1 = max(ya1, yb1)
    xi2 = min(xa2, xb2)
    yi2 = min(ya2, yb2)
    iw = max(0.0, xi2 - xi1)
    ih = max(0.0, yi2 - yi1)
    inter = iw * ih
    a_area = max(0.0, xa2 - xa1) * max(0.0, ya2 - ya1)
    b_area = max(0.0, xb2 - xb1) * max(0.0, yb2 - yb1)
    union = a_area + b_area - inter + 1e-9
    return inter / union if union > 0 else 0.0

def compute_ap(recs, precs):
    mrec = np.concatenate(([0.0], recs, [1.0]))
    mpre = np.concatenate(([0.0], precs, [0.0]))
    for i in range(mpre.size-2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i+1])
    idx = np.where(mrec[1:] != mrec[:-1])[0]
    ap = 0.0
    for i in idx:
        ap += (mrec[i+1] - mrec[i]) * mpre[i+1]
    return ap

def simple_iou_stats(decoded_boxes, gt_labels_batch, iou_threshold=0.5):
    best_ious = []
    above_count = 0
    total = len(decoded_boxes)
    for i, preds in enumerate(decoded_boxes):
        gts = []
        lab = gt_labels_batch[i]
        cell_size = img_size / grid_size
        for gy in range(grid_size):
            for gx in range(grid_size):
                for a in range(anchor_count):
                    if lab[gy,gx,a,4] > 0.5:
                        tx,ty,tw,th = lab[gy,gx,a,0:4]
                        cx = (gx + tx)  * cell_size
                        cy = (gy + ty) * cell_size
                        w = math.exp(tw) * anchors[a][0]
                        h = math.exp(th) * anchors[a][1]
                        x1,y1,x2,y2 = cx-w/2, cy-h/2, cx+w/2, cy+h/2
                        gts.append([x1,y1,x2,y2])
        if len(preds) == 0 or len(gts) == 0:
            best_ious.append(0.0)
            continue
        preds_sorted = sorted(preds, key=lambda x: x[4], reverse=True)
        best_pred = preds_sorted[0][:4]
        ious = [iou_np(best_pred, gt) for gt in gts]
        best = max(ious) if ious else 0.0
        best_ious.append(best)
        if best >= iou_threshold:
            above_count += 1
    mean_best = float(np.mean(best_ious)) if total > 0 else 0.0
    frac = above_count / total if total > 0 else 0.0
    return {'mean_best_iou': mean_best, 'frac_above_threshold': frac, 'per_image_iou': best_ious }

def build_gt_and_pred_lists(label_paths, decoded_all, img_size=img_size, grid_size=grid_size, anchors=anchors, anchor_count=anchor_count):

    gt_by_image = []
    for lp in label_paths:
        lab = np.load(lp)
        gts = []
        cell_size = img_size / grid_size
        for gy in range(grid_size):
            for gx in range(grid_size):
                for a in range(anchor_count):
                    if lab[gy,gx,a,4] > 0.5:
                        tx,ty,tw,th = lab[gy,gx,a,0:4]
                        cx = (gx + tx) * cell_size
                        cy = (gy + ty) * cell_size
                        w = math.exp(tw) * anchors[a][0]
                        h = math.exp(th) * anchors[a][1]
                        x1,y1,x2,y2 = cx - w/2.0, cy - h/2.0, cx + w/2.0, cy + h/2.0
                        class_vec = lab[gy,gx,a,5:]
                        class_id = int(np.argmax(class_vec)) if class_vec.sum() > 0 else 0
                        gts.append([x1, y1, x2, y2, class_id])
        gt_by_image.append(gts)
    pred_by_image = decoded_all  
    return gt_by_image, pred_by_image

def evaluate_model(model, image_paths, label_paths, max_images=None):
    ips = image_paths[:max_images] if max_images else image_paths
    lps = label_paths[:max_images] if max_images else label_paths
    decoded_all = []
    gt_all = []
    for ip, lp in zip(ips, lps):
        img = tf.io.read_file(ip)
        img = tf.image.decode_jpeg(img, channels=3)
        img = tf.image.resize(img, (img_size, img_size))
        img = tf.cast(img, tf.float32) / 255.0
        img_np = img.numpy() if hasattr(img, "numpy") else np.array(img)
        pred = model.predict(np.expand_dims(img_np, 0), verbose=0)
        dets = decode_predictions(pred)
        decoded_all.append(dets)
        gt = np.load(lp)
        gt_all.append(gt)

    stats = simple_iou_stats(decoded_all, np.stack(gt_all, axis=0), iou_threshold=0.5)
    return stats

def evaluate_map(model, image_paths, label_paths, max_images=None):
    ips = image_paths[:max_images] if max_images else image_paths
    lps = label_paths[:max_images] if max_images else label_paths
    decoded_all = []
    for ip in ips:
        img = tf.io.read_file(ip)
        img = tf.image.decode_jpeg(img, channels=3)
        img = tf.image.resize(img, (img_size, img_size))
        img = tf.cast(img, tf.float32) / 255.0
        img_np = img.numpy() if hasattr(img, "numpy") else np.array(img)
        pred = model.predict(np.expand_dims(img_np, 0), verbose=0)
        dets = decode_predictions(pred)
        decoded_all.append(dets)

    gt_by_image, pred_by_image = build_gt_and_pred_lists(lps, decoded_all)

    res_ap = voc_ap_from_preds(gt_by_image, pred_by_image, num_classes=num_classes, iou_thresh=0.5)

    conf_thresh = 0.5
    pr_at_conf = precision_recall_at_confidence(gt_by_image, pred_by_image, num_classes=num_classes, conf_thresh=conf_thresh, iou_thresh=0.5)

    result = {
        "AP_results": res_ap,
        "pr_at_conf": pr_at_conf,
        "conf_thresh": conf_thresh
    }

    print("mAP@0.5:", res_ap["mAP"])
    for c in range(num_classes):
        p = pr_at_conf[c]["precision"]
        r = pr_at_conf[c]["recall"]
        tp = pr_at_conf[c]["tp"]
        fp = pr_at_conf[c]["fp"]
        fn = pr_at_conf[c]["fn"]
        print(f"Class {c}: precision@{conf_thresh}={p:.3f}, recall@{conf_thresh}={r:.3f} (TP={tp} FP={fp} FN={fn})")

    return result

def visualize_predictions(images, decoded_boxes, class_names=None, figsize=(6,6)):
    n = len(decoded_boxes)
    for i in range(n):
        img = images[i]
        fig, ax = plt.subplots(1, figsize=figsize)
        ax.imshow((img * 255).astype(np.uint8) if img.dtype == np.float32 else img.astype(np.uint8))
        for box in decoded_boxes[i]:
            x1,y1,x2,y2, score,cls = box
            rect = patches.Rectangle((x1,y1), x2-x1, y2-y1, linewidth=2, edgecolor='r', facecolor='none')
            ax.add_patch(rect)
            label = f"{class_names[cls] if class_names else cls}:{score:.2f}"
            ax.text(x1, max(0,y1-6), label, color='yellow', fontsize=8, bbox=dict(facecolor='black', alpha=0.6, pad=1))
        ax.axis('off')
        fig.tight_layout()
        plt.show()

def visualize_with_gt(images, decoded_boxes, gt_labels_batch, class_name=None):
    cell_size = img_size / grid_size
    for i in range(len(decoded_boxes)):
        img = images[i]
        fig, ax = plt.subplots(1, figsize=(6,6))
        ax.imshow((img * 255).astype(np.uint8) if img.dtype == np.float32 else img.astype(np.uint8))
        for box in decoded_boxes[i]:
            x1,y1,x2,y2,score,cls = box
            rect = patches.Rectangle((x1,y1), x2-x1, y2-y1, linewidth=2, edgecolor='r', facecolor='none')
            ax.add_patch(rect)
            ax.text(x1, max(0,y1-6), f"P{cls}:{score:.2f}", color='yellow', fontsize=8, bbox=dict(facecolor='black',alpha=0.6))
        lab = gt_labels_batch[i]
        for gy in range(grid_size):
            for gx in range(grid_size):
                for a in range(anchor_count):
                    if lab[gy,gx,a,4] > 0.5:
                        tx,ty,tw,th = lab[gy,gx,a,0:4]
                        cx = (gx + tx) * cell_size
                        cy = (gy + ty) * cell_size
                        w = math.exp(tw) * anchors[a][0]
                        h = math.exp(th) * anchors[a][1]
                        x1,y1,x2,y2 = cx-w/2, cy-h/2, cx+w/2, cy+h/2
                        rect = patches.Rectangle((x1,y1), x2-x1, y2-y1, linewidth=2, edgecolor='g', facecolor='none')
                        ax.add_patch(rect)
        ax.axis('off')
        fig.tight_layout()
        plt.show()

def main():
    model = create_yolo_model()
    model.summary()
    loss_fn = yolo_loss()
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss=loss_fn)

    image_paths = np.load("paths_images.npy", allow_pickle=True).tolist()
    label_paths = np.load("paths_labels.npy", allow_pickle=True).tolist()

    N = len(image_paths)
    idx = np.arange(N)
    np.random.shuffle(idx)
    n_train = int(N * 0.8)
    n_val   = int(N * 0.1)
    train_idx = idx[:n_train]
    val_idx   = idx[n_train:n_train + n_val]
    test_idx  = idx[n_train + n_val:]

    train_images = [image_paths[i] for i in train_idx]
    train_labels = [label_paths[i] for i in train_idx]
    val_images   = [image_paths[i] for i in val_idx]
    val_labels   = [label_paths[i] for i in val_idx]
    test_images  = [image_paths[i] for i in test_idx]
    test_labels  = [label_paths[i] for i in test_idx]

    train_ds = create_dataset(train_images, train_labels, batch_size=batch_size, training=True)
    val_ds = create_dataset(val_images, val_labels, batch_size=batch_size, training=False)
    test_ds = create_dataset(test_images, test_labels, batch_size=batch_size, training=False)

    ckpt_dir = os.path.expanduser("models")
    os.makedirs(ckpt_dir, exist_ok=True)
    ckpt_path = os.path.join(ckpt_dir, "yolo_epoch_{epoch:02d}_valLoss_{val_loss:.4f}.keras")

    checkpoint_cb = tf.keras.callbacks.ModelCheckpoint(
        filepath=ckpt_path, monitor="val_loss", save_best_only=True, verbose=1
    )

    earlystop_cb = tf.keras.callbacks.EarlyStopping( monitor='val_loss', patience=5, restore_best_weights=True, verbose=1)
    callbacks = [checkpoint_cb, earlystop_cb]

    model.fit(train_ds, validation_data=val_ds, epochs=epochs, callbacks=callbacks)
    
    # Save final model
    #model.save(os.path.join(ckpt_dir, "yolo_mink_recognition_v0.keras"))
    #print("Saved model")

if __name__ == "__main__":
    main()
