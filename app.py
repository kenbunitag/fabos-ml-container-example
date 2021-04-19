import ml_models
from flask import Flask, flash, request, redirect, url_for, send_file, render_template
app = Flask(__name__)
import os
from io import BytesIO
import torch
import aioredis
import asyncio
import msgpack
import cv2
from PIL import Image
import sys
import base64

TORCH_HOME = "<not set>"
if "TORCH_HOME" in os.environ:
    TORCH_HOME = os.environ['TORCH_HOME']
    print(f"using TORCH_HOME={os.environ['TORCH_HOME']}")
resnet18 = ml_models.Resnet18Wrapper()
mask_rcnn = ml_models.MaskRCNNWrapper()


REDIS_HOST = os.environ.get("REDIS_HOST", "localhost")
REDIS_PORT = os.environ.get("REDIS_PORT", "6379")

REDIS_STREAM_IN = os.environ.get("REDIS_STREAM_IN", "mask_rcnn-requests")
REDIS_STREAM_OUT = os.environ.get("REDIS_STREAM_IN", "mask_rcnn-results")

loop = asyncio.get_event_loop()

@app.route('/')
def main():
    return render_template("main.html", 
        cuda_is_available=torch.cuda.is_available(), 
        TORCH_HOME=TORCH_HOME, 
        mask_rcnn_dump=repr(mask_rcnn.model),
        resent18_dump=repr(resnet18.model)
        )


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

def serve_pil_image(pil_img):
    img_io = BytesIO()
    pil_img.save(img_io, 'JPEG', quality=70)
    img_io.seek(0)
    return send_file(img_io, mimetype='image/jpeg')
    
@app.route('/maskrcnn', methods=['GET', 'POST'])
def maskrcnn_demo():
    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        image, masks, boxes, pred_cls = mask_rcnn.instance_segmentation_api(file.stream)
        # if user does not select file, browser also
        # submit an empty part without filename
        #print("saving ot tmp.jpg")
        #file.save("tmp.jpg")
        #print("do_inference tmp.jpg")
        #image = mask_rcnn.instance_segmentation_api("tmp.jpg")
        return serve_pil_image(image)
        #return send_file("mask_rcnn_result.png")
        
    return '''
    <!doctype html>
    <title>MaskRCNN Demo</title>
    <h1>MaskRCNN Demo</h1>
    <form method=post enctype=multipart/form-data>
      <input type=file name=file>
      <input type=submit value=Upload>
    </form>
    '''

async def redis_request_response(image_stream):
    r : aioredis.Redis = await aioredis.create_redis(f'redis://{REDIS_HOST}:{REDIS_PORT}/0')
    id = await r.xadd(REDIS_STREAM_IN, dict(_transfer="inline", data=msgpack.packb(dict(image=image_stream.read()))))
    response_stream = f"{REDIS_STREAM_OUT}-{id}"
    result = await r.xread([response_stream], 5000, count=1, latest_ids=["0"])
    for e in result:
        stream, id, msg = e
        data = msgpack.unpackb(msg[b"data"])
        return data

async def redis_read_all():
    r : aioredis.Redis = await aioredis.create_redis(f'redis://{REDIS_HOST}:{REDIS_PORT}/0')
    result = await r.xread([REDIS_STREAM_OUT], 1, count=None, latest_ids=["0"])
    datas = []
    for e in result:
        stream, id, msg = e
        #print(msg.keys(), file=sys.stdout)
        response_stream = msg[b"stream"].decode("UTF-8")
        inner_result = await r.xread([response_stream], 1, count=1, latest_ids=["0"])
        for ee in inner_result:
            stream, id, msg = ee
            print(msg.keys(), file=sys.stdout)
            #return msg.keys()
            data = msgpack.unpackb(msg[b"data"])
            print("data.keys()", data.keys(), file=sys.stdout)
            data["response_stream"] = response_stream
            datas.append(data)
    return datas


@app.route('/remote', methods=['GET', 'POST'])
def remote_demo():
    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        
        data = loop.run_until_complete(redis_request_response(file))
        #img = ml_models.get_opencv_img_from_buffer(BytesIO(data["image"]), cv2.IMREAD_ANYCOLOR)
        #im_pil = Image.fromarray(img)

        # ignore and just display all previcous results
        #return serve_pil_image(im_pil)
        
    datas = loop.run_until_complete(redis_read_all())

    response = []
    datas = reversed(datas)
    for data in datas:
        print(type(data), file=sys.stdout)
        image_bytes = data["image"]
        response.append(f"response stream: {data['response_stream']}<br />")
        response.append(f"boxes: {data['boxes']}<br />")
        response.append(f"pred_cls: {data['pred_cls']}<br />")
        response.append(f"<img src=\"data:image/png;base64, {base64.b64encode(image_bytes).decode('UTF-8')}\"  /><br />")
    return '''
    <!doctype html>
    <title>Remote Demo</title>
    <h1>Remote Demo</h1>
    <form method=post enctype=multipart/form-data>
      <input type=file name=file>
      <input type=submit value=Upload>
    </form>
    <h2>Previous requests</h2>
    ''' + "".join(response)

@app.route('/banana', methods=['GET', 'POST'])
def infer():
    return resnet18.do_inference("images/Banana-Single.jpg")

if __name__ == '__main__':
    app.run(host='0.0.0.0', port='5000', debug=True)