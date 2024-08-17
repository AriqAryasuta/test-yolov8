import sys
sys.path.insert(0, "/kaggle/working/test-yolov8/ultralytics")
# 现在就可以导入Yolo类了
from ultralytics import YOLO

# Load a model
# model = YOLO('/home/jiayuan/yolom/ultralytics/models/v8/yolov8-bdd-v4-one-dropout-individual-n.yaml', task='multi')  # build a new model from YAML
# model = YOLO('yolov8n.pt')  # load a pretrained model (recommended for training)
model = YOLO('/kaggle/working/test-yolov8/ultralytics/models/v8/yolov8-bdd-v4-one-dropout-individual-s.yaml', task='detect').load('/kaggle/input/vehicle/pytorch/default/1/v4s.pt')  # build from YAML and transfer weights

# Train the model

model.train(data='/kaggle/working/test-yolov8/ultralytics/datasets/bdd-multi.yaml', batch=4, epochs=300, imgsz=(640,640), device=[0], name='yolopm', val=True,classes=[2,6],combine_class=[],single_cls=False)

