from ultralytics import YOLO

model = YOLO("yolov8m.pt")

model.export(
    format="coreml",
    imgsz=640,
    nms=True,
    half=True
)

print("Export complete! yolov8m.mlpackage is ready.")
