from ultralytics import YOLO

# Load a model
model = YOLO("yolov8n.yaml")  # build a new model from scratch

# Use the model
result = model.train(data=".\dataset\data.yaml", epochs=100)
# train the model
