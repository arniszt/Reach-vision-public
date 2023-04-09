import torch
import torchvision

import fiftyone as fo
import fiftyone.zoo as foz

from PIL import Image
from torchvision.transforms import functional as func
from torchvision.models import resnet50, ResNet50_Weights

import cv2

import torchvision.transforms as transforms
import numpy as np
import json
import requests
import matplotlib.pyplot as plt


from matplotlib import image as mpimg
 



import coco_labels as cl

#import warnings
from Setupdatamodel import dataset 

from fiftyone import ViewField as F

# Run the model on GPU if it is available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Load a pre-trained Faster R-CNN model

#pre-trained IMAGENET1K
#model = resnet50(weights=ResNet50_Weights.DEFAULT)
#pre-trained COCO

model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True) 
#model = torch.hub.load('facebookresearch/semi-supervised-ImageNet1K-models', 'resnet50_swsl')
print(dir(model))



#resnet50 = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_resnet50', pretrained=True)
#model = resnet50
#utils = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_convnets_processing_utils')

#print("Hola")
#canviant aquesta linea canviariem el model
model.to(device)
#print("Hello")

model.eval()

#print("Model ready")

#print(dataset)


cl.set_coco_labels()



print(cl.coco_labels)

#sample = dataset.first()
#print(sample.ground_truth.detections[0])
#predictions_view = dataset.take(100, seed=51)
predictions_view = dataset.take(1,51)

classes = dataset.default_classes

pb = fo.ProgressBar()

sample = list(pb(predictions_view))[0]
# Add predictions to samples
#with fo.ProgressBar() as pb:
#    for sample in pb(predictions_view):
        # Load image
image = Image.open(sample.filepath)
image = func.to_tensor(image).to(device)
c, h, w = image.shape

        # Perform inference
preds = model([image])[0]
        
        #img=Image.open(sample.filepath)
        #img.show()

labels = preds["labels"].cpu().detach().numpy()
scores = preds["scores"].cpu().detach().numpy()
boxes = preds["boxes"].cpu().detach().numpy()

print(boxes)

for label, score, box in zip(preds["labels"], preds["scores"], preds["boxes"]):
    if score > 0.75 and label < len(cl.coco_labels): 
        # Get the name of the label
        print(f'Label {cl.coco_labels[label]}, score {score}, box {box}')

        # Convert detections to FiftyOne format
detections = []
for label, score, box in zip(labels, scores, boxes):
    # Convert to [top-left-x, top-left-y, width, height]
    # in relative coordinates in [0, 1] x [0, 1]
    x1, y1, x2, y2 = box
    rel_box = [x1 / w, y1 / h, (x2 - x1) / w, (y2 - y1) / h]

    detections.append( 
    fo.Detection(
        label=classes[label],
        bounding_box=rel_box,
        confidence=score
        )
    )

    # Save predictions to dataset
sample["faster_rcnn"] = fo.Detections(detections=detections)
sample.save()
img = cv2.imread(sample.filepath)
# get contours
result = img.copy()

for label, score, box in zip(labels, scores, boxes): 
    if score > 0.75 and label < len(cl.coco_labels):
        # Convert to [top-left-x, top-left-y, width, height]
        # in relative coordinates in [0, 1] x [0, 1] 
            x0 = list(box)[0]
            y0 = list(box)[1]
            x1 = list(box)[2]
            y1 = list(box)[3]
            start_point = (int(x0), int(y0))
            end_point = (int(x1), int(y1))
            print(cl.coco_labels[label])
            
            
            text = cl.coco_labels[label]
            font = cv2.FONT_HERSHEY_SIMPLEX
            org = (int(x0), int(y0))
            fontScale = 0.5
            color = (0, 0, 255)
            thickness = 2
            result = cv2.putText(result, text, org, font, fontScale, color, thickness, cv2.LINE_AA, False) 
          
          
            cv2.rectangle(result, start_point, end_point, (0, 0, 255), 2)

            
cv2.imshow("bounding_box", result)
cv2.waitKey(0)
cv2.destroyAllWindows()
'''
print("Finished adding predictions")
session = fo.launch_app(dataset)
session.view = predictions_view
results = predictions_view.evaluate_detections(
    "faster_rcnn",
    gt_field = "ground_truth",
    eval_key = "eval", 
)


#high_conf_view=predictions_view.filter_labels("faster_rcnn", F("confidence") > 0.75, only_matches=False)
#print(high_conf_view)

#session.view = high_conf_view


#results = high_conf_view.evaluate_detections(
#    "faster_rcnn",
#    gt_field = "ground_truth",
#    eval_key = "eval", 
#)

session.wait()'''