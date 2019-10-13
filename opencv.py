import numpy as np
import cv2
import datetime

# fungsi untuk menulis text
def makeText(text='demo', bg_color=(255, 0, 0), color=(255, 255, 255)):
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
	
# haarscade milik opencv
bodyPath = 'haarcascade/haarcascade_upperbody.xml'
eyePath = 'haarcascade/haarcascade_eye_tree_eyeglasses.xml'
cascPath = 'haarcascade/haarcascade_frontalface_default.xml'

# deklarasi classifier
faceCascade = cv2.CascadeClassifier(cascPath)
eyesCascade = cv2.CascadeClassifier(eyePath)
bodyCascade = cv2.CascadeClassifier(bodyPath)

# ambil video dari web-cam, mulai dari 0
# kalu ingin url ganti angka 0 dengan IP
video_capture = cv2.VideoCapture(0)

# proses video
while video_capture:

	# baca frame
	ret, frame = video_capture.read()
	
	# ambil ukuran frame untuk menaruh label
	height, width, channels = frame.shape
	
	# ubah warna frame menjadi gray
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	
	
	# prediksi body
	bodies = bodyCascade.detectMultiScale(
		gray,
		scaleFactor=1.1,
		minNeighbors=5,
		minSize=(30,30),
	)
	
	# draw
	for(x, y, w, h) in bodies:
		# beri kotak pada objek
		cv2.rectangle(frame, (x,y), (x+w, y+h), (255,0,255), 2)
	
	# end of prediksi body
		
	# prediksi face
	faces = faceCascade.detectMultiScale(
		gray,
		scaleFactor=1.1,
		minNeighbors=5,
		minSize=(30,30),
	)
	
	# draw
	for(fx, fy, fw, fh) in faces:
		
		# beri kotak pada objek
		cv2.rectangle(frame, (fx,fy), (fx+fw, fy+fh), (0,255,0), 2)
		
		# ambil gambar dari prediksi muka untuk prediksi mata
		roi_gray = gray[fy:fy+fh, fx:fx+fw]
		roi_color = frame[fy:fy+fh, fx:fx+fw]
		
		# prediksi mata
		eyes = eyesCascade.detectMultiScale(roi_gray)
		
		for(ex, ey, ew, eh) in eyes:
		
			# beri kotak pada objek
			cv2.rectangle(roi_color, (ex,ey), (ex+ew, ey+eh), (255,255,0), 2)
	# end of prediksi muka
	
	# hitung body yang terdeteksi
	people = np.array(bodies).shape
	
	# timestamp
	time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
	
	# variabel untuk menampung jumlah orang
	people_number = str(0)
	
	# jika lebih dari 1 maka diupdate labelnya
	if people[0] > 0:
		people_number = str(people[0])
	
	print(people)
	
	# label text yang akan ditampilkan
	text = time+" \n People : "+people_number
	
	# pasang label
	makeText(text)
	
	# tampilkan video
	cv2.imshow('Video', frame)
	
	# stop
	if cv2.waitKey(1) & 0xFF == ord('q'):
		break
		
# keluar 
video_capture.release()
cv2.destroyAllWindows()