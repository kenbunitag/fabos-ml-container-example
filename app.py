import torch
import torchvision
import torchvision.models as models
from PIL import Image
from torchvision import transforms
import numpy as np
from omegaconf import OmegaConf

from flask import Flask, flash, request, redirect, url_for
app = Flask(__name__)

model = models.resnet18(pretrained=True)
label_names = OmegaConf.load("labels.yaml")["labels"]
print(f"loaded {len(label_names)} label names")

def do_inference(path):
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
        model.to('cuda')

    with torch.no_grad():
        output = model(input_batch)
    # Tensor of shape 1000, with confidence scores over Imagenet's 1000 classes
    #print(output[0]) 
    # The output has unnormalized scores. To get probabilities, you can run a softmax on it.
    probs = torch.nn.functional.softmax(output[0], dim=0)
    top_probs = torch.topk(probs, k=5)
    matching_labels = [label_names[i.item()] for i in top_probs.indices]
    return(dict(values=top_probs.values.tolist(), indices=top_probs.indices.tolist(), names=matching_labels))

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        # if user does not select file, browser also
        # submit an empty part without filename
        print("saving ot tmp.jpg")
        file.save("tmp.jpg")
        print("do_inference tmp.jpg")
        return do_inference("tmp.jpg")
        
    return '''
    <!doctype html>
    <title>Upload jpg File</title>
    <h1>Upload jpg File</h1>
    <form method=post enctype=multipart/form-data>
      <input type=file name=file>
      <input type=submit value=Upload>
    </form>
    '''

@app.route('/banana', methods=['GET', 'POST'])
def infer():
    return do_inference("images/Banana-Single.jpg")

if __name__ == '__main__':
    app.run(host='localhost', port='5000', debug=True)