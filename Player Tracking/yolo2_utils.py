import numpy as np
import argparse
import cv2 as cv
import subprocess
import time
import os


def draw_boxes(img, boxes, confidences, classids, idxs, colors, labels):
    if len(idxs) > 0:
        for i in idxs.flatten():
            x, y = boxes[i][0], boxes[i][1]
            w, h = boxes[i][2], boxes[i][3]
    
            color = [int(c) for c in colors[classids[i]]]


            cv.rectangle(img, (x, y), (x+w, y+h), color, 2)
            text = labels[classids[i]]
            cv.putText(img, text, (x, y-5), cv.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    return img


def generate_boxes(outs, height, width, tconf):
    boxes = []
    confidences = []
    classids = []

    for out in outs:
        for detection in out:
            scores = detection[5:]
            classid = np.argmax(scores)
            confidence = scores[classid]
            
            if confidence > tconf:

                box = detection[0:4] * np.array([width, height, width, height])
                centerX, centerY, bwidth, bheight = box.astype('int')


                x = int(centerX - (bwidth / 2))
                y = int(centerY - (bheight / 2))


                boxes.append([x, y, int(bwidth), int(bheight)])
                confidences.append(float(confidence))
                classids.append(classid)

    return boxes, confidences, classids

def infer_image(net, layer_names, height, width, img, colors, labels, boxes=None, confidences=None, classids=None, idxs=None, infer=True):
    
    if infer:

        blob = cv.dnn.blobFromImage(img, 1 / 255.0, (416, 416), swapRB=True, crop=False)


        net.setInput(blob)
        outs = net.forward(layer_names)
       
        boxes, confidences, classids = generate_boxes(outs, height, width, 0.5)
        

        idxs = cv.dnn.NMSBoxes(boxes, confidences, 0.5, 0.3)

    if boxes is None or confidences is None or idxs is None or classids is None:
        raise 'error'
        
    # Draw labels and boxes on the image
    img = draw_boxes(img, boxes, confidences, classids, idxs, colors, labels)

    return img, boxes, confidences, classids, idxs
