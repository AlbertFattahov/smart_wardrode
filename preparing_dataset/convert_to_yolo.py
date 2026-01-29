import os
from tqdm import tqdm
import yaml
from PIL import Image


ROOT = r"C:\Users\Альберт\Downloads\Category and Attribute Prediction Benchmark"
IMG_DIR = os.path.join(ROOT, "img")
ANNO_DIR = os.path.join(ROOT, "Anno_coarse")
EVAL_FILE = os.path.join(ROOT, "Eval", "list_eval_partition.txt")

OUT = r"C:\IT\Python\Smart_wardrode\output_yolo"

os.makedirs(os.path.join(OUT, "images", "train"), exist_ok=True)
os.makedirs(os.path.join(OUT, "images", "val"), exist_ok=True)
os.makedirs(os.path.join(OUT, "labels", "train"), exist_ok=True)
os.makedirs(os.path.join(OUT, "labels", "val"), exist_ok=True)

#загрузка категорий
cat_map={}
with open(os.path.join(ANNO_DIR, "list_category_cloth.txt")) as f: 
    lines = f.readlines()[2:] 
    for i, line in enumerate(lines, start=1): 
        name, cat_type = line.strip().split() 
        cat_map[i] = name


#перенумерация категорий(было с 1, надо с 0)
cat_to_yolo = {cid: i for i, cid in enumerate(cat_map.keys())}

#загрузка bbox 
bbox_map = {}
with open(os.path.join(ANNO_DIR, "list_bbox.txt")) as f:
    lines = f.readlines()[2:]
    for line in lines:
        parts = line.split()
        img = parts[0]
        x1, y1, x2, y2 = map(int, parts[1:])
        bbox_map[img] = (x1, y1, x2, y2)

#загрузка категорий изображений
img_cat = {}
with open(os.path.join(ANNO_DIR, "list_category_img.txt")) as f:
    lines = f.readlines()[2:]
    for line in lines:
        img, cid = line.split()
        img_cat[img] = int(cid)


#загрузка train\val split
split = {}
with open(EVAL_FILE) as f:
    lines = f.readlines()[2:]
    for line in lines:
        parts = line.split() 
        if len(parts) != 2: continue 
        img, part = parts 
        split[img] = part

    
#Конвертация в YOLO
for img in tqdm(bbox_map.keys()):
    if img not in img_cat:
        continue

    x1, y1, x2, y2 = bbox_map[img]
    cid = img_cat[img]
    yolo_cid = cat_to_yolo[cid]

    img_path = os.path.join(ROOT, img)
    im = Image.open(img_path)
    w, h = im.size

    xc = (x1 + x2)/2 / w
    yc = (y1 + y2)/2 / h
    bw = (x2 - x1)/ w
    bh = (y2 - y1)/ h

    part = split.get(img, "train")
    part = "train" if part == "train" else "val"
    

    #filename = os.path.basename(img) 
    filename = img.replace("/", "_")
    out_img_path = os.path.join(OUT, "images", part, filename)
    im.save(out_img_path)

    label_path = os.path.join(OUT, "labels", part, filename.replace(".jpg", ".txt"))
    with open(label_path, "w") as f:
        f.write(f"{yolo_cid} {xc:.6f}  {yc:.6f} {bw:.6f} {bh:.6f}\n")


#Создание data.yaml

data = {
    "train": os.path.join(OUT, "images/train"),
    "val": os.path.join(OUT, "images/val"),
    "ns": len(cat_map),
    "name": list(cat_map.values())
}

with open(os.path.join(OUT, "data.yaml"), "w") as f:
    yaml.dump(data, f)

print("Готово! УСПЕХ")




