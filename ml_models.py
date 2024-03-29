import torch
import torchvision
import torchvision.models as models
from PIL import Image
from torchvision import transforms
import numpy as np
from omegaconf import OmegaConf
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import torchvision.transforms as T
import matplotlib.pyplot as plt
import cv2
import random
import time
import math


def get_opencv_img_from_buffer(buffer, flags):
    bytes_as_np_array = np.frombuffer(buffer.read(), dtype=np.uint8)
    return cv2.imdecode(bytes_as_np_array, flags)
    
class Resnet18Wrapper:
    def __init__(self):
        self.label_names = OmegaConf.load("labels.yaml")["labels"]
        print(f"loaded {len(self.label_names)} label names")
        self.model = models.resnet18(pretrained=True)


    def do_inference(self, path):
        input_image = Image.open(path)
        print(input_image)
        preprocess = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        input_tensor = preprocess(input_image)
        input_batch = input_tensor.unsqueeze(0) # create a mini-batch as expected by the model

        # move the input and model to GPU for speed if available
        if torch.cuda.is_available():
            input_batch = input_batch.to('cuda')
            self.model.to('cuda')

        with torch.no_grad():
            output = self.model(input_batch)
        # Tensor of shape 1000, with confidence scores over Imagenet's 1000 classes
        #print(output[0]) 
        # The output has unnormalized scores. To get probabilities, you can run a softmax on it.
        probs = torch.nn.functional.softmax(output[0], dim=0)
        top_probs = torch.topk(probs, k=5)
        matching_labels = [label_names[i.item()] for i in top_probs.indices]
        return(dict(values=top_probs.values.tolist(), indices=top_probs.indices.tolist(), names=matching_labels))



class MaskRCNNWrapper:
    def __init__(self):
        # load a model pre-trained pre-trained on COCO
        #model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
        self.model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)
        self.model.eval()


    COCO_INSTANCE_CATEGORY_NAMES = [
        '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
        'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign',
        'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
        'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A', 'N/A',
        'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
        'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
        'bottle', 'N/A', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
        'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
        'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table',
        'N/A', 'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
        'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A', 'book',
        'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
    ]


    def get_prediction(self, img, threshold):
        #img = Image.open(img_path)
        transform = T.Compose([T.ToTensor()])
        img = transform(img)
        start = time.time()
        pred = self.model([img])
        end = time.time()
        print(f"model inference duration: {end - start}")
        pred_score = list(pred[0]['scores'].detach().numpy())
        pred_t = [pred_score.index(x) for x in pred_score if x > threshold][-1]
        masks = (pred[0]['masks'] > 0.5).squeeze().detach().cpu().numpy()
        pred_class = [self.COCO_INSTANCE_CATEGORY_NAMES[i]
                    for i in list(pred[0]['labels'].numpy())]
        pred_boxes = [[(i[0], i[1]), (i[2], i[3])]
                    for i in list(pred[0]['boxes'].detach().numpy())]
        masks = masks[:pred_t+1]
        pred_boxes = pred_boxes[:pred_t+1]
        pred_class = pred_class[:pred_t+1]
        return masks, pred_boxes, pred_class


    def random_colour_masks(self, image):
        colours = [[0, 255, 0], [0, 0, 255], [255, 0, 0], [0, 255, 255], [255, 255, 0], [
            255, 0, 255], [80, 70, 180], [250, 80, 190], [245, 145, 50], [70, 150, 250], [50, 190, 190]]
        r = np.zeros_like(image).astype(np.uint8)
        g = np.zeros_like(image).astype(np.uint8)
        b = np.zeros_like(image).astype(np.uint8)
        r[image == 1], g[image == 1], b[image == 1] = colours[random.randrange(0, 10)]
        coloured_mask = np.stack([r, g, b], axis=2)
        return coloured_mask


    def instance_segmentation_api(self, img_path, threshold=0.5, rect_th=3, text_size=3, text_th=3, return_type="PIL") -> (Image, np.ndarray, list, list):
        if isinstance(img_path, str):
            img = cv2.imread(img_path)
        else:
            img = get_opencv_img_from_buffer(img_path, cv2.IMREAD_ANYCOLOR)
        
        scale_factor = min(1000 / img.shape[0], 1000 / img.shape[1])
        if scale_factor < 1:
            width = int(img.shape[1] * scale_factor)
            height = int(img.shape[0] * scale_factor)
            dim = (width, height)
            img = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
        
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        im_pil = Image.fromarray(img)

        masks, boxes, pred_cls = self.get_prediction(im_pil, threshold)

        if return_type == "PIL":
            for i in range(len(masks)):
                rgb_mask = self.random_colour_masks(masks[i])
                img = cv2.addWeighted(img, 1, rgb_mask, 0.5, 0)
                cv2.rectangle(img, boxes[i][0], boxes[i][1],
                            color=(0, 255, 0), thickness=rect_th)
                cv2.putText(img, pred_cls[i], boxes[i][0], cv2.FONT_HERSHEY_SIMPLEX,
                            text_size, (0, 255, 0), thickness=text_th)
            im_pil = Image.fromarray(img)
            #cv2.imwrite("mask_rcnn_result.png", img)
            return (im_pil, masks, boxes, pred_cls)

        #plt.figure(figsize=(20, 30))
        #plt.imshow(img)
        #plt.xticks([])
        #plt.yticks([])
        #plt.savefig("mask_rcnn_result.png")
        #plt.show()

