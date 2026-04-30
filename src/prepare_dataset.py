import os
import json
from pathlib import Path
import numpy as np
from PIL import Image
from mink_recognition import encode_labels, img_size

def prepare_data(root_dir, anno_json, out_label_dir, save_paths_file_prefix="paths"):
    os.makedirs(out_label_dir, exist_ok=True)
    data = json.load(open(anno_json, 'r'))
    image_paths = []
    label_paths = []

    for item in data:
        rel = item.get('file')
        if rel is None:
            continue
        img_path = os.path.join(root_dir, rel)
        if not os.path.exists(img_path):
            print("Missing image:", img_path)
            continue

        img = Image.open(img_path).convert('RGB')
        ow, oh = img.size

        boxes_xywh = []
        classes = []
        for (x1,y1,x2,y2), cls in zip(item.get('boxes', []), item.get('classes', [])):
            if 0.0 <= x1 <= 1.0 and 0.0 <= x2 <= 1.0 and 0.0 <= y1 <= 1.0 and 0.0 <= y2 <= 1.0:
                x1 = x1 * ow
                x2 = x2 * ow
                y1 = y1 * oh
                y2 = y2 * oh

            x1s = x1 * img_size / ow
            x2s = x2 * img_size / ow
            y1s = y1 * img_size / oh
            y2s = y2 * img_size / oh

            cx = (x1s + x2s) / 2.0
            cy = (y1s + y2s) / 2.0
            w = x2s - x1s
            h = y2s - y1s
            boxes_xywh.append([cx, cy, w, h])
            classes.append(int(cls))

        if len(boxes_xywh) == 0:
            y = np.zeros((13,13,3,5+4), dtype=np.float32)  
        else:
            y = encode_labels(np.array(boxes_xywh), np.array(classes))

        base = Path(rel).stem
        label_path = os.path.join(out_label_dir, base + ".npy")
        np.save(label_path, y.astype(np.float32))

        image_paths.append(os.path.abspath(img_path))
        label_paths.append(os.path.abspath(label_path))

    np.save(save_paths_file_prefix + "_images.npy", np.array(image_paths))
    np.save(save_paths_file_prefix + "_labels.npy", np.array(label_paths))
    print("Prepared:", len(image_paths), "items. Labels in", out_label_dir)

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--root", required=True, help="root dir for images")
    p.add_argument("--anno", required=True, help="annotations json path")
    p.add_argument("--out", required=True, help="output label dir")
    args = p.parse_args()
    prepare_data(args.root, args.anno, args.out)
