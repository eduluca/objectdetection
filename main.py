import cv2

#img = cv2.imread("lena.PNG")

#Opencv for videocapture feature
cap = cv2.VideoCapture(0)
cap.set(3,640)
cap.set(4,480)

#Setting up the class names for object detection
classNames = []
classFile = 'coco.names'
with open(classFile,'rt') as f:
    classNames = f.read().rstrip('\n').split('\n')

#Configuration and weights set up
configPath = 'ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
weightsPath = 'frozen_inference_graph.pb'

#Detection model set up
net = cv2.dnn_DetectionModel(weightsPath,configPath)
net.setInputSize(320,320)
net.setInputScale(1.0/27.5)
net.setInputMean((127.5,127.5,127.5))
net.setInputSwapRB(True)

#Loop for real time object detection
while True:
    success, img = cap.read()
    classIds, confs, bbox = net.detect(img,confThreshold=0.6) #Only show if 60% or more confident
    print(classIds,bbox)
    with open(classFile, 'rt') as f:
        classNames = [line.rstrip() for line in f] #to fix classID-1 error
    if len(classIds) != 0:
        for classId, confidence,box in zip(classIds.flatten(),confs.flatten(),bbox):
            cv2.rectangle(img,box,color=(0,255,0),thickness=2) #green box preferences
            cv2.putText(img,classNames[classId - 1],(box[0] + 10, box[1] + 30),
                        cv2.FONT_HERSHEY_COMPLEX, 1, (0,255,0),2) #green name preferences


        cv2.imshow("Output",img)
        cv2.waitKey(1)

    


