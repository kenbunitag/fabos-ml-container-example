import ml_models
from flask import Flask, flash, request, redirect, url_for, send_file
app = Flask(__name__)

resnet18 = ml_models.Resnet18Wrapper()
mask_rcnn = ml_models.MaskRCNNWrapper()


@app.route('/')
def main():
    return '''
    <!doctype html>
    <title>ML Inference demo</title>
    <h1>ML Inference demo</h1>
    <a href="/resnet18">ResNet18 Demo</a><br />
    <a href="/maskrcnn">MaskRCNN Demo</a><br />
    '''

@app.route('/resnet18', methods=['GET', 'POST'])
def resnet18_demo():
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
        return resnet18.do_inference("tmp.jpg")
        
    return '''
    <!doctype html>
    <title>ResNet18 Demo</title>
    <h1>ResNet18 Demo</h1>
    <form method=post enctype=multipart/form-data>
      <input type=file name=file>
      <input type=submit value=Upload>
    </form>
    '''

@app.route('/maskrcnn', methods=['GET', 'POST'])
def maskrcnn_demo():
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
        mask_rcnn.instance_segmentation_api("tmp.jpg")
        return send_file("mask_rcnn_result.png")
        
    return '''
    <!doctype html>
    <title>MaskRCNN Demo</title>
    <h1>MaskRCNN Demo</h1>
    <form method=post enctype=multipart/form-data>
      <input type=file name=file>
      <input type=submit value=Upload>
    </form>
    '''

@app.route('/banana', methods=['GET', 'POST'])
def infer():
    return resnet18.do_inference("images/Banana-Single.jpg")

if __name__ == '__main__':
    app.run(host='0.0.0.0', port='5000', debug=True)