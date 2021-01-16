import numpy as np
import cv2 as cv
from yolo2_utils import infer_image


if __name__ == '__main__':
	labels = open('./yolov3-coco/coco-labels').read().strip().split('\n')

	colors = np.random.randint(0, 255, size=(len(labels), 3), dtype='uint8')

	net = cv.dnn.readNetFromDarknet('./yolov3-coco/yolov3.cfg', './yolov3-coco/yolov3.weights')

	layer_names = net.getLayerNames()
	layer_names = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
        
	if True:
		try:
			vid = cv.VideoCapture('3.mp4')
			height, width = None, None
			writer = None
		except:
			raise 'error'

		finally:
			while True:
				flag, frame = vid.read()

				if not flag:
					break

				if width is None or height is None:
					height, width = frame.shape[:2]

				frame, box, _, _, _ = infer_image(net, layer_names, height, width, frame, colors, labels)
				print(box[0])
				if writer is None:
					fourcc = cv.VideoWriter_fourcc(*"MJPG")
					writer = cv.VideoWriter('./output2.avi', fourcc, 30, (frame.shape[1], frame.shape[0]), True)

				writer.write(frame)
			print ("DONE")
			writer.release()
			vid.release()


