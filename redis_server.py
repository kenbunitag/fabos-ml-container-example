import aioredis
import os
import asyncio
import ml_models
from io import BytesIO
import cv2
from PIL import Image
import msgpack
import numpy as np

REDIS_HOST = os.environ.get("REDIS_HOST", "localhost")
REDIS_PORT = os.environ.get("REDIS_PORT", "6379")

REDIS_STREAM_IN = os.environ.get("REDIS_STREAM_IN", "mask_rcnn-requests")
REDIS_STREAM_OUT = os.environ.get("REDIS_STREAM_IN", "mask_rcnn-results")

if "TORCH_HOME" in os.environ:
    print(f"using TORCH_HOME={os.environ['TORCH_HOME']}")

async def main():
    r : aioredis.Redis = await aioredis.create_redis(f'redis://{REDIS_HOST}:{REDIS_PORT}/0')
    timeout = 0
    latest_id = "0"
    await r.delete(REDIS_STREAM_IN)
    await r.delete(REDIS_STREAM_OUT)

    mask_rcnn = ml_models.MaskRCNNWrapper()

    img = cv2.imread("images/Intersection.jpg")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    im_pil = Image.fromarray(img)
    img_io = BytesIO()
    im_pil.save(img_io, 'JPEG', quality=70)
    img_io.seek(0)
    

    await r.xadd(REDIS_STREAM_IN, dict(_transfer="inline", data=msgpack.packb(dict(image=img_io.read()))))


    while True:
        print(f"waiting for requests on stream {REDIS_STREAM_IN}")

        result = await r.xread([REDIS_STREAM_IN], timeout, count=1, latest_ids=[latest_id])
        for e in result:
            stream, id, msg = e
            id = id.decode("UTF-8")
            print(f"got request {id}")

            data = msgpack.unpackb(msg[b"data"])
            #print(data.keys())
            img_bytes = BytesIO(data["image"])
            image, masks, boxes, pred_cls = mask_rcnn.instance_segmentation_api(img_bytes, return_type="PIL")
            img_io = BytesIO()
            image.save(img_io, 'JPEG', quality=70)
            img_io.seek(0)
            img_bytes = img_io.read()

            masks=np.array(masks).tolist()
            boxes=np.array(boxes).tolist()

            response_stream = f"{REDIS_STREAM_OUT}-{id}"

            response_id = await r.xadd(response_stream, dict(_transfer="inline", request_id=id, data=msgpack.packb(dict(image=img_bytes, masks=masks, boxes=boxes, pred_cls=pred_cls))))
            await r.xadd(REDIS_STREAM_OUT, dict(_transfer="stream", reply_to=id, stream=response_stream))
            print(f"produced result on {response_stream} with id {response_id}")

            latest_id = id



asyncio.run(main())