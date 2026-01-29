from ultralytics import YOLO
import json
import os

MODEL_PATH = "models/exp19/weights/best.pt" #  путь к модели
SOURCE_DIR = "test_images" # изображения которые нужно обработать
OUT_JSON = "results.json" # путь(название) файла для сохранения


model = YOLO(MODEL_PATH)
results = model(SOURCE_DIR, imgsz=640, conf=0.3, save=False) # save=True если нужно сохранить фотографии с отрисованными bbox

out = []

for r in results:
    try:
        image_name = os.path.basename(r.path)
    except Exception:
        image_name = None

    for box in r.boxes:
        cls_id = int(box.cls[0])

        if hasattr(model, "names"): class_name = model.names[cls_id]
        else:
            class_name = str(cls_id)

        rec = {
            "name_file": image_name,
            "ID" : cls_id,
            "class_ name": class_name
        }
        out.append(rec)

with open( OUT_JSON, "w", encoding="utf-8") as f:
    json.dump(out, f, indent=2)

print(f"Сохранено {len(out)} записей в {OUT_JSON}")