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
    x2 = cx - w/2.0
    y2 = cy - h/2.0
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

# Sett opp en YOLO modell
def create_yolo_model(input_shape=(img_size, img_size, 3), num_classes=num_classes, anchors=anchors): 
    
    # Input layer
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

    # output layer
    anchor_count = anchors.shape[0]
    output_filters = anchor_count * (5 + num_classes)
    outputs = layers.Conv2D(output_filters, kernel_size=1, padding="same")(x)

    model = models.Model(inputs, outputs)
    return model

# Encode labels -> grid (numpy)
def encode_labels(boxes, classes, img_size=img_size, grid_size=grid_size, anchors=anchors, num_classes=num_classes):
    """
    boxes: (N,4) in absolute pixel coords [cx,cy,w,h]
    classes: (N,) ints
    returns y: (grid_size,grid_size,anchor_count,5+num_classes)
    """
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
        ty = (cx / cell_size) - gy
        tw = math.log((w + 1e-9) / (anchors[best_idx][0] + 1e-9))
        th = math.log((h + 1e-9) / (anchors[best_idx][1] + 1e-9))

        y[gy, gx, best_idx, 0:4] = [tx, ty, tw, th]
        y[gy, gx, best_idx, 4] = 1.0
        y[gy, gx, best_idx, 5 + cls] = 1.0
    return y

# Dummy data generator
def generate_dummy_data(num_images):

    images = np.random.rand(num_images, img_size, img_size, 3).astype(np.float32)

    labels = np.zeros((num_images, grid_size, grid_size, anchor_count, 5 + num_classes), dtype=np.float32)

    for i in range(num_images):
        nobj = np.random.randint(1,4)
        boxes = []
        classes = []
        for _ in range(nobj):
            w = np.random.uniform(20,50)
            h = np.random.uniform(20,50)
            cx = np.random.uniform(w/2, img_size - w/2)
            cy = np.random.uniform(h/2, img_size - h/2)
            cls = np.random.randint(0, num_classes)
            boxes.append([cx,cy,w,h])
            classes.append(cls)
        labels[i] = encode_labels(np.array(boxes), np.array(classes))
    return images, labels

# YOLO loss
def yolo_loss(anchors=anchors, grid_size=grid_size, num_classes=num_classes):
    anchor_count = anchors.shape[0]
    #bce = tf.keras.losses.BinaryCrossentropy(from_logits=True, reduction=tf.keras.losses.Reduction.NONE) 
    cce = tf.keras.losses.CategoricalCrossentropy(from_logits=True, reduction=tf.keras.losses.Reduction.NONE) 

    def loss(y_true, y_pred):
        # Splitt prediksjoner
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

# Dataset pipeline 
def create_dataset(images, labels, batch_size=batch_size, training=True, shuffle_buffer=None):
    dataset = tf.data.Dataset.from_tensor_slices((images, labels))
    
    if training:
        if shuffle_buffer is None:
            shuffle_buffer = min(1000, len(images))
        dataset = dataset.shuffle(buffer_size=shuffle_buffer)
        
        # map augment etter shuffle så hver augment får variert input
        dataset = dataset.map(lambda x, y: (augment(x), y),
                              num_parallel_calls=tf.data.AUTOTUNE)
        
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    return dataset

def softmax(x):
    e = np.exp(x - np.max(x))
    return e / e.sum()

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

# Inference helpers: decode predictions, NMS...
def decode_predictions(pred, img_size=img_size, grid_size=grid_size, anchors=anchors, num_classes=num_classes): 
    """
    pred: (grid,grid,anchors*(5+num_classes)) or (1,grid,grid,anchors*(...))
    returns list of boxes [(x1,y1,x2,y2,score,class_id), ...] in absolute pixel coords
    """
    if pred.ndim == 4: 
        pred = np.squeeze(pred, axis=0)
    # reshape
    pred = pred.reshape((grid_size, grid_size, anchor_count, 5 + num_classes))
    cell_size = img_size
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
                # decode
                cx = (gx + sigmoid(tx) * cell_size)
                cy = (gy + sigmoid(ty) * cell_size)
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
    # NSM per class
    final = []
    for c in range(num_classes):
        cls_boxes = [b for b in boxes if b[5] == c]
        if not cls_boxes:
            continue
        # sort by score
        cls_boxes.sort(key=lambda  x: x[4], reverse=True)
        kept = []
        while cls_boxes:
            box = cls_boxes.pop(0)
            kept.append(box)
            cls_boxes = [b for b in cls_boxes if iou_np(box[:4], b[:4]) < nms_iou_threshold]
        final.extend(kept)
    return final

def visualize_predictions(images, decoded_boxes, class_names=None, figsize=(6,6)):
    n = len(decoded_boxes)
    for i in range(n):
        img = images[i]
        print("VISUALIZE image dtype/min/max:", img.dtype, img.min(), img.max())
        fig, ax = plt.subplots(1, figsize=figsize)
        ax.imshow((img * 255).astype(np.uint8) if img.dtype == np.float32 else img.astype(np.uint8))
        for box in decoded_boxes[i]:
            x1,y1,x2,y2, score,cls = box
            w = x2 - x1
            h = y2 - y1
            rect = patches.Rectangle((x1,y1), w, h, linewidth=2, edgecolor='r', facecolor='none')
            ax.add_patch(rect)
            label = f"{class_names[cls] if class_names else cls}:{score:.2f}"
            ax.text(x1, max(0,y1-6), label, color='yellow', fontsize=8,
                    bbox=dict(facecolor='black', alpha=0.6, pad=1))
        ax.axis('off')
        fig.tight_layout()
        plt.show()

def visualize_with_gt(images, decoded_boxes, gt_labels_batch, class_name):
    # gt_labels_batch: array (N, G, G, A, 5+C)
    cell_size = img_size / grid_size
    for i in range(len(decoded_boxes)):
        img = images[i]
        fig, ax = plt.subplots(1, figsize=(6,6))
        ax.imshow((img * 255).astype(np.uint8) if img.dtype == np.float32 else img.astype(np.uint8))
        # draw preds (rød)
        for box in decoded_boxes[i]:
            x1,y1,x2,y2,score,cls = box
            rect = patches.Rectangle((x1,y1), x2-x1, y2-y1, linewidth=2, edgecolor='r', facecolor='none')
            ax.add_patch(rect)
            ax.text(x1, max(0,y1-6), f"P{cls}:{score:.2f}", color='yellow', fontsize=8, bbox=dict(facecolor='black',alpha=0.6))
        # draw GTs (grønn)
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

# Evaluation 
def evalute_model(model, dataset_images, dataset_labels, iou_threshold=0.5):
    decoed_all = []
    for i, img in enumerate(dataset_images):
        inp = np.expand_dims(img, 0)
        pred = model.predict(inp, verbose=0) 
        dets = decode_predictions(pred)
        decoed_all.append(dets)
    stats = simple_iou_stats(decoed_all, dataset_labels, iou_threshold)
    return {'mean_best_iou': stats['mean_best_iou'], 'frac_iou>0.5': stats['frac_above_threshold']}

# Training main
def main():
    
    model = create_yolo_model()
    model.summary()

    loss_fn = yolo_loss()
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss=loss_fn)

    # Create dummy dataset (replace with real)
    train_images, train_labels = generate_dummy_data(100)
    val_images, val_labels = generate_dummy_data(20)

    # Sjekk numpy‑arrays (min/maks)
    print("TRAIN images dtype/min/max:", train_images.dtype, train_images.min(), train_images.max())
    print("VAL   images dtype/min/max:", val_images.dtype, val_images.min(), val_images.max())

    train_ds = create_dataset(train_images, train_labels, batch_size, training=True)
    val_ds = create_dataset(val_images, val_labels, batch_size, training=False)

    ckpt_dir = r"C:\Users\Kevin\Documents\Hobbyprosjekt4 AI med mink\Models"
    os.makedirs(ckpt_dir, exist_ok=True)
    ckpt_path = os.path.join(ckpt_dir, "yolo_epoch_{epoch:02d}_valLoss_{val_loss:.4f}.keras")

    checkpoint_cb = tf.keras.callbacks.ModelCheckpoint(
        filepath=ckpt_path,
        monitor="val_loss",
        save_best_only=True,
        verbose=1
    )

    earlystop_cb = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=5,
        restore_best_weights=True,
        verbose=1
    )

    callbacks = [checkpoint_cb, earlystop_cb]

    for x_batch, y_batch in train_ds.take(1):
        # x_batch er en tf.Tensor; bruk .numpy() for å få numpy‑verdier i eager mode
        print("BATCH x_batch dtype:", x_batch.dtype)
        print("BATCH x_batch min/max:", x_batch.numpy().min(), x_batch.numpy().max())
        print("BATCH y_batch dtype:", y_batch.dtype)
        print("BATCH y_batch min/max (objectness channel):", y_batch.numpy()[...,4].min(), y_batch.numpy()[...,4].max())
        # fortsatt i debug: gjør én forward pass
        y_pred = model(x_batch, training=False)
        _ = loss_fn(y_batch, y_pred)
        break

    model.fit(
        train_ds, 
        validation_data=val_ds, 
        epochs=epochs, 
        callbacks=callbacks
    )
    
    x = val_images[:4]  # 4 eksempelbilder
    preds = model.predict(x)
    print("preds shape:", preds.shape)

    p0 = preds[0].reshape((grid_size, grid_size, anchor_count, 5 + num_classes))
    print("raw pred (cell 6,6 anchor0 first 6):", p0[6,6,0,:6])

    # b) decode & visualise
    decoded_list = []
    for i in range(len(x)):
        p = preds[i:i+1]  # keep batch dim for decode function
        dets = decode_predictions(p)
        decoded_list.append(dets)
    visualize_predictions(x, decoded_list, class_names=[str(i) for i in range(num_classes)])

    # c) simple numeric IoU sanity check (best pred per image)
    stats = simple_iou_stats(decoded_list, val_labels[:4], iou_threshold=0.5)
    print("Simple IoU stats (first 4 images):")
    print(" Mean best IoU:", stats['mean_best_iou'])
    print(" Fraction above IoU>=0.5:", stats['frac_above_threshold'])
    print(" Per-image IoUs:", stats['per_image_iou'])

    # Save final model
    #save_dir = r"C:\Users\Kevin\Documents\Hobbyprosjekt4 AI med mink\Models"
    #os.makedirs(save_dir, exist_ok=True)

    # Lag et filnavn 
    #save_path = os.path.join(save_dir, "yolo_mink_recognition_v0.keras")
    #model.save(save_path)
    #print("Saved model to:", save_path)

if __name__ == "__main__":
    main()
