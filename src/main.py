from yolo_runner import YoloRunner

def main():
    MODEL_PATH = "models/exp19/weights/best.pt"
    SOURCE_DIR = "test_images"
    OUT_JSON = "results.json"

    runner = YoloRunner(MODEL_PATH, imgsz=640, conf=0.3)

    data = runner.run(SOURCE_DIR, save_images=False)

    runner.save(data, OUT_JSON)


if __name__ == "__main__":
    main()
