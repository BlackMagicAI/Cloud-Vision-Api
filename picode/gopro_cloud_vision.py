import numpy as np
import time
from PIL import Image, ImageDraw, ImageFont
from picamera import PiCamera
from picamera.array import PiRGBArray
import base64
import json
from io import BytesIO
import os
import subprocess
# Using Python Request library
import requests

#########################################
# File:   gopro_cloud_vision.py
# GoPro camera object detection using tiny-yolo 3 model.
# Author: Maurice Tedder
# Date:   Nov. 4, 2021
##Ref:
##  https://www.pyimagesearch.com/2015/03/30/accessing-the-raspberry-pi-camera-with-opencv-and-python
##  https://picamera.readthedocs.io/en/release-1.13/recipes1.html#overlaying-images-on-the-preview
##  https://picamera.readthedocs.io/en/release-1.13/api_camera.html#module-picamera
##  https://pillow.readthedocs.io/en/stable/reference/Image.html#PIL.Image.fromarray
##  https://github.com/raspberrypilearning/the-all-seeing-pi/blob/master/code/overlay_functions.py

### Start of gcloud authentication code - uncomment if you have gcloud installed
### Follow the instruction in the following link to install gcloud on a Linux 32-bit (x86) platform
### Raspberry PI. https://cloud.google.com/sdk/docs/install
# # uncomment the following code if you have gcloud installed
# # set system environment variable
# os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = '<INSERT-CREDENTIALS-FILE-PATH-HERE>'
# # get bearer token from command line gcloud
# bearer_token=subprocess.run(["gcloud", "auth", "application-default", "print-access-token"],stdout=subprocess.PIPE, universal_newlines=True)
# time.sleep(2)
# BEARER_TOKEN = bearer_token.stdout.strip()
### End of gcloud authentication code

## Paste BEARER_TOKEN from goole Colaboratory here. Comment this code if you are using gcloud
BEARER_TOKEN = '<INSERT-BEARER-TOKEN-COPIED-FROM-NOTEBOOK-HERE>'

# define variables
url = 'https://vision.googleapis.com/v1/images:annotate'
bearer = 'Bearer {}'.format(BEARER_TOKEN)

######Start function definitions
# Pass the image data to an encoding function.
def encode_image(image):
  infile = BytesIO()
  image.save(infile, format="JPEG")
  return base64.b64encode(infile.getvalue()).decode('utf-8')

# Draw bounding box, class labels and scores onto original image
def drawBoundingBox(image, boxes, all_classes, scores, color):
  fnt =ImageFont.truetype('./fonts/Ubuntu-Bold.ttf', 18)
  # fnt = ImageFont.load_default()
  draw = ImageDraw.Draw(image)
  for box, cl, score in zip(boxes, classes, scores):
    p0, p1, p2, p3 = box
    x0, y0 = p0
    x1, y1 = p2

    top = x0 * image.width
    left = y0 * image.height
    right = x1 * image.width
    bottom = y1 * image.height

    draw.rectangle([(top, left), (right, bottom)], outline=color)
    draw.text((top + 4, left), '{0} {1:.2f}'.format(cl, score), font=fnt, fill=(0, 0, 255))

# Process json response
def processJson(response):
  boxes = []
  classes = []
  scores = []
  for responseObject in response.get('responses'):
    for localizedObjectAnnotation in responseObject.get('localizedObjectAnnotations', []):
      classes.append(localizedObjectAnnotation.get('name'))
      scores.append(localizedObjectAnnotation.get('score'))
      box = []
      for normalizedVertice in localizedObjectAnnotation.get('boundingPoly').get('normalizedVertices'):
        if (normalizedVertice.get('x') != None and normalizedVertice.get('y') != None):
          box.append((normalizedVertice['x'], normalizedVertice['y']))
        else:
          box.clear()
          break

      if (bool(box)):
        boxes.append(box)
  return boxes, classes, scores

# Form JSON object for POST request to Vision API.
def getJSONBody(image):
  return {"requests": [
      {
        "image": {
          "content": encode_image(image)
        },
        "features": [
          {
            "maxResults": 10,
            "type": "OBJECT_LOCALIZATION"
          }
        ]
      }
    ]
  }

#Used to print time lapsed during different parts of code execution.
def printTimeStamp(start, log):
  end = time.time()
  print(log + ': time: {0:.2f}s'.format(end-start))

# Combine original image & bounding box annotation overlays image into one image
# & save to jpg file.
# From Ref: https://github.com/raspberrypilearning/the-all-seeing-pi/blob/master/code/overlay_functions.py
def output_overlay(filepath, output=None, overlay=None):

  # Take an overlay Image
  overlay_img = overlay.convert('RGBA')

  # ...and a captured photo
  output_img = output.convert('RGBA')

  # Combine the two and save the image as output
  new_output = Image.alpha_composite(output_img, overlay_img)
  new_output.save(filepath, "JPEG")

######End function definitions

# Create camera & rawcapture objects
#stream = BytesIO()
camera = PiCamera()
rawCapture = PiRGBArray(camera, size=camera.resolution)
print(camera.resolution)
#camera.start_preview(fullscreen=False, window=(0,0,size[0],size[1])) #show preview in custom size. Used for debugging.
camera.start_preview(fullscreen=True)

pad = None

for frame in camera.capture_continuous(rawCapture, format='rgb'):

  image = Image.fromarray(frame.array)

  # Create an image padded to the required size with
  # mode 'RGBA' needed for bounding box overlay with transparency mask.
  pad = Image.new('RGBA', (
    ((image.size[0] + (32-1)) // 32) * 32,
    ((image.size[1] + (16-1)) // 16) * 16,
    ))

  #Program reference start timestamp
  start = time.time()

  # Form json request payload
  jsonBody = getJSONBody(image)

  # send POST request to url
  response = requests.post(url,
                    headers={'Content-Type': 'application/json; charset=utf-8',
                             'Authorization': bearer},
                    json=jsonBody
                    )

  # printTimeStamp(start, "Detection Time")

  #Process model output data annotate image with bounding boxes
  boxes, classes, scores = processJson(response.json())
  drawBoundingBox(pad, boxes, classes, scores, (0,255,0))

  #Remove previous overlays
  for o in camera.overlays:
    camera.remove_overlay(o)

  o = camera.add_overlay(pad.tobytes(), alpha = 255, layer = 3, size=pad.size)

  rawCapture.truncate(0)
  #break #uncomment to end loop and output image to jpg file

# Combine original image & bounding box annotation overlays image into one image
# & save to jpg file.
output_overlay("out/goproyolo.jpg", image, pad)

camera.stop_preview()
camera.close()
