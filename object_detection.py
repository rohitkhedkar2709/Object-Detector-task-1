import cv2
import numpy as np

net = cv2.dnn.readNet('yolov3.weights' , 'yolov3.txt')
classes = []
with open('coco.txt', 'r') as f:
    classes = f.read().splitlines()

#to  load the image
img = cv2.imread('img3.jpg')
print("Object detection is performing")

height, width, _ = img.shape
#constract blob for the image for fixed height and width
blob = cv2.dnn.blobFromImage(img, 1/255, (416, 416), (0, 0, 0), swapRB=True, crop=False)
#to set the input to the network and compute forward
# pass for the input and storing result as layerOutputs
net.setInput(blob)
output_layers_names = net.getUnconnectedOutLayersNames()
layerOutputs = net.forward(output_layers_names)

boxes = []
confidences = []
class_ids = []
#loop for layerOutputs
for output in layerOutputs:
    for detection in output:
        scores = detection[5:]
        class_id = np.argmax(scores)
        confidence = scores[class_id]
        if confidence > 0.5:
            center_x = int(detection[0]*width)
            center_y = int(detection[1]*height)
            w = int(detection[2]*width)
            h = int(detection[3]*height)
            x = int(center_x - w/2)
            y = int(center_y - h/2)

            boxes.append([x, y, w, h])
            confidences.append((float(confidence)))
            class_ids.append(class_id)

indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
font = cv2.FONT_HERSHEY_PLAIN
#to give random colours for bounding boxes
colors = np.random.uniform(0, 255, size=(len(boxes), 3))

for i in indexes.flatten():
    x, y, w, h = boxes[i]
    label = str(classes[class_ids[i]])
    confidence = str(round(confidences[i], 2))
    color = colors[i]
    cv2.rectangle(img, (x, y), (x+w, y+h), color, 2)
    cv2.putText(img, label + " " + confidence, (x, y+20), font, 2, (255, 255, 255), 2)
#dispaly the image
cv2.imshow('Image', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
