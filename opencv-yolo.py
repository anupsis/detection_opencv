import cv2
import numpy as np

# Load Yolo
net = cv2.dnn.readNet("cfg/yolov3.weights", "cfg/yolov3.cfg")
classes = []
with open("data/coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]
	
layer_names = net.getLayerNames()
output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
colors = np.random.uniform(0, 255, size=(len(classes), 3))

import numpy as np
import cv2
import datetime

eyePath = 'haarcascade/haarcascade_eye_tree_eyeglasses.xml'
facecPath = 'haarcascade/haarcascade_frontalface_default.xml'

faceCascade = cv2.CascadeClassifier(facecPath)
eyesCascade = cv2.CascadeClassifier(eyePath)

def makeText(frame, text='demo', bg_color=(255, 0, 0), color=(255, 255, 255)):
	pos = (0, (height-25))
	font_face = cv2.FONT_HERSHEY_SIMPLEX
	scale = 0.5
	thickness = cv2.FILLED
	margin = 30
	
	txt_size = cv2.getTextSize(text, font_face, scale, thickness)
	end_x = pos[0] + txt_size[0][0] + margin
	end_y = pos[1] - txt_size[0][1] - margin
	
	# cv2.rectangle(frame, pos, (end_x, end_y), bg_color, thickness)
	for i, line in enumerate(text.split('\n')):
		z = pos[1]+(i*15)
		cv2.putText(frame,line, (pos[0], z), font_face, scale, color, 1, 2)
	
video_capture = cv2.VideoCapture(0)
while True:
	ret, img = video_capture.read()
	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	# height, width, channels = frame.shape
	# gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	
	# Loading image
	# img = frame
	# img = cv2.resize(img, None, fx=0.4, fy=0.4)
	height, width, channels = img.shape

	# Detecting objects
	blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
	net.setInput(blob)
	outs = net.forward(output_layers)

	# Showing informations on the screen
	class_ids = []
	confidences = []
	boxes = []
	for out in outs:
		for detection in out:
			scores = detection[5:]
			class_id = np.argmax(scores)
			confidence = scores[class_id]
			# if confidence > 0.5 and class_id == 2:
			if confidence > 0.5 and class_id == 0:
				# Object detected
				center_x = int(detection[0] * width)
				center_y = int(detection[1] * height)
				w = int(detection[2] * width)
				h = int(detection[3] * height)
				# Rectangle coordinates
				x = int(center_x - w / 2)
				y = int(center_y - h / 2)
				boxes.append([x, y, w, h])
				confidences.append(float(confidence))
				class_ids.append(class_id)
				
				# for face
				froi_gray = gray[y:y+h, x:x+w]
				froi_color = img[y:y+h, x:x+w]
				
				faces = faceCascade.detectMultiScale(
					froi_gray,
					scaleFactor=1.1,
					minNeighbors=5,
					minSize=(30,30),
				)
				
				for(fx, fy, fw, fh) in faces:
					cv2.rectangle(froi_color, (fx,fy), (fx+fw, fy+fh), (0,255,0), 2)
					
					roi_gray = gray[fy:fy+fh, fx:fx+fw]
					roi_color = img[fy:fy+fh, fx:fx+fw]
					
					eyes = eyesCascade.detectMultiScale(roi_gray)
					
					for(ex, ey, ew, eh) in eyes:
						cv2.rectangle(roi_color, (ex,ey), (ex+ew, ey+eh), (255,255,0), 2)
				
				

	indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

	font = cv2.FONT_HERSHEY_PLAIN
	for i in range(len(boxes)):
		if i in indexes:
			x, y, w, h = boxes[i]
			label = str(classes[class_ids[i]])
			print(label)
			print(class_ids[i])
			color = colors[i]
			cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
			
	time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
	text = time+" \n People : "+str(len(class_ids))
	makeText(img, text)
	
	cv2.imshow('Video', img)
	
	if cv2.waitKey(1) & 0xFF == ord('q'):
		break
		
video_capture.release()
cv2.destroyAllWindows()