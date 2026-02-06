from ultralytics import YOLO
import json
import os

class YoloRunner:
    def __init__(self, model_path: str, imgsz: int = 640, conf: float = 0.3):
        self.model = YOLO(model_path)
        self.imgsz = imgsz
        self.conf = conf
        print(f"Model {os.path.basename(model_path)} initialized.")

    def run(self, source_dir: str, save_images: bool = False):
        results = self.model(
            source_dir,
            imgsz=self.imgsz,
            conf=self.conf,
            save=save_images
        )

        out = []
        
        for r in results:
            try:
                image_name = os.path.basename(r.path)
            except Exception:
                image_name = None

            for box in r.boxes:
                cls_id = int(box.cls[0])

                if hasattr(self.model, "names"):
                    class_name = self.model.names[cls_id]
                else:
                    class_name = str(cls_id)

                out.append({
                    "name_file": image_name,
                    "ID": cls_id, 
                    "class_name": class_name
                })

        return out
    
    def save(self, data, out_json: str):
        with open(out_json, "w", encoding="utf-8") as f:
            json.dump(data, f, indent = 3)

        print(f"Сохранено {len(data)} записей в {out_json}")