from matplotlib.style import use
import torch
import cv2
import argparse
# torch.cuda.is_available()
import time
from model.ultrafastLaneDetector import UltrafastLaneDetector, ModelType

parse = argparse.ArgumentParser()

parse.add_argument('-i', '--image', type=str)
parse.add_argument('-v', '--video', type=str)
parse.add_argument('-c', '--confidence', type=int, default=0.3)
parse.add_argument('-w', '--weights', type=str, default='./model/weight/best.pt')

args = vars(parse.parse_args())
fps_start_time = 0
list_fps = []
fps = 0

model_path = "./model/weight/culane_18.pth"
model_type = ModelType.CULANE
use_gpu = False

vid = cv2.VideoCapture(args['video'])

lane_detector = UltrafastLaneDetector(model_path, model_type, use_gpu)
writer = None
model = torch.hub.load('D:\Python\ADAS\Yolov5\MQsolution\model\yolov5_master','custom', path=args['weights'],force_reload=True,source='local')
# try:
while(True):
                fps_end_time = time.time()
                time_diff = fps_end_time - fps_start_time
                fps = 1/time_diff
                list_fps.append(fps)
                fps_start_time = fps_end_time
                if len(list_fps) % 10 == 0:
                        print("FPS: ",sum(list_fps)/len(list_fps))

                ret, img = vid.read()
                img = lane_detector.detect_lanes(img)
                frame = cv2.resize(img, (640, 640))

                detected = model(frame)
        
                results = detected.pandas().xyxy[0].to_dict(orient="records")
                for result in results:
                                confid = result['confidence']
                                label = result['name']
                                if confid > args['confidence']:
                                        x1 = int(result['xmin'])
                                        y1 = int(result['ymin'])
                                        x2 = int(result['xmax'])
                                        y2 = int(result['ymax'])
                                        w = int(x1+(x2-x1)/2)
                                        h = int(y1+(y2-y1)/2)
                                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                                        cv2.putText(frame, label, (int(x1), int(y1)), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)
                                        # cv2.putText(frame, str(confid), (int(x1), int(y1)), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2) 

                cv2.imshow("Window", frame)
                if writer is None:
                        fourcc = cv2.VideoWriter_fourcc(*"MJPG")
                        writer = cv2.VideoWriter('./output.avi', fourcc, 30, (frame.shape[0], frame.shape[1]), True)
                writer.write(frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
vid.release()
cv2.destroyAllWindows()