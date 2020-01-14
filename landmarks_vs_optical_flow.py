import numpy as np
import cv2
import argparse
import imutils
import dlib
import face_recognition


path_to_shape_predictor = r"./shape_predictor_68_face_landmarks.dat"
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(path_to_shape_predictor)

def optical_flow(old_gray, frame_gray, p0):
	lk_params = dict( winSize  = (15,15),maxLevel = 2,criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
	p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)
	if st is not None:
		good_new = p1[st==1]
		good_old = p0[st==1]
		return good_new, good_old
	else:
		return None, None

flag = True
writer = None
cap = cv2.VideoCapture(0)

while(1):
	if flag:
		flag =False
		ret, old_frame = cap.read()
		old_frame = imutils.resize(old_frame, width=500)
		old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
		faces = detector(old_gray, 1)
		for face in faces:
			shape = predictor(old_gray, face)
			p0 = np.float32(np.asarray([[(shape.part(30).x, shape.part(30).y)], [(shape.part(36).x, shape.part(36).y)],[(shape.part(45).x, shape.part(45).y)], [(shape.part(48).x, shape.part(48).y)], [(shape.part(54).x, shape.part(54).y)], [(shape.part(8).x, shape.part(8).y)]]))
			mask = np.zeros_like(old_frame)

	ret,frame = cap.read()
	frame = imutils.resize(frame, width=500)
	frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

	faces_lk = detector(frame_gray, 1)
	for face in faces_lk:
		shape_lk = predictor(frame_gray, face)
		shape_lk_coordinates = [[shape_lk.part(30).x, shape_lk.part(30).y], [shape_lk.part(36).x, shape_lk.part(36).y],[shape_lk.part(45).x, shape_lk.part(45).y], [shape_lk.part(48).x, shape_lk.part(48).y], [shape_lk.part(54).x, shape_lk.part(54).y], [shape_lk.part(8).x, shape_lk.part(8).y]]
		for x,y in shape_lk_coordinates:
			frame = cv2.circle(frame,(x, y),  2, (0, 255, 255), 3)
	
	good_new, good_old = optical_flow(old_gray, frame_gray, p0)
	for i,(new,old) in enumerate(zip(good_new, good_old)):
		a,b = new.ravel()
		c,d = old.ravel()
		mask = cv2.line(mask, (a,b),(c,d), (0, 0, 255),1)
		frame = cv2.circle(frame,(a,b),  1, (0, 0, 255), 2)
	cv2.putText(frame, "Optical Flow Result", (30,15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
	cv2.circle(frame,(20,15),  1, (0, 255, 255), 3)
	cv2.putText(frame, "Facial Landmark Result", (30,35), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
	cv2.circle(frame,(20,35),  1, (0, 0, 255), 2)
	cv2.imshow('frame',frame)
	if writer is None:
		fourcc = cv2.VideoWriter_fourcc(*"MJPG")
		writer = cv2.VideoWriter("landmarks_VS_optical_flow.avi", fourcc, 10,
			(frame.shape[1], frame.shape[0]), True)
	if writer is not None:
		writer.write(frame)	

	key = cv2.waitKey(1) & 0xFF
	if key == 27:
		break
	old_gray = frame_gray.copy()
	p0 = good_new.reshape(-1,1,2)
cv2.destroyAllWindows()
cap.release()
