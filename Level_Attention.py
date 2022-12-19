"""Demo code showing how to estimate human head pose.

There are three major steps:
1. Detect human face in the video frame.
2. Run facial landmark detection on the face image.
3. Estimate the pose by solving a PnP problem.

To find more details, please refer to:
https://github.com/yinguobing/head-pose-estimation
"""
from argparse import ArgumentParser

import cv2
import time
import csv


from mark_detector import MarkDetector
from pose_estimator import PoseEstimator
from fuzzylogic.classes import Domain, Rule #fuzzy
from fuzzylogic.functions import R, S, triangular, trapezoid #fuzzy

#FUZZY PART
#tune this values according to the ones desired for the membership functions
#set the ranges for the membership functions and resolution
HR = Domain("Horizontal Rotation", -1.5, 1.5, res=0.001)
#tune this values according to the ones desired for the membership functions
HR.left = S(-0.5, 0)
HR.middle = trapezoid(-0.5,0,0.3,0.5)
HR.right = R(0.3,0.5)
#uncomment to print the membership functions
#HR.left.plot()
#uncomment to print the membership functions
#HR.middle.plot()
#uncomment to print the membership functions
#HR.right.plot()

#set the ranges for the membership functions and resolution
VR = Domain("Vertical Rotation", -0.85, 0.85, res=0.001)
#tune this values according to the ones desired for the membership functions
VR.up = S(0, 0.15)
VR.middle = trapezoid(0,0.15,0.45,0.6)
VR.down = R(0.45,0.6)
#uncomment to print the membership functions
#VR.up.plot()
#uncomment to print the membership functions
#VR.middle.plot()
#uncomment to print the membership functions
#VR.down.plot()

#set the ranges for the membership functions and resolution
D = Domain("Depth", -1300, -100, res=0.001)
#tune this values according to the ones desired for the membership functions
D.far = S(-1060, -750)
D.optimal = triangular(-1060,-440)
D.close = R(-750, -440)
#uncomment to print the membership functions
#D.far.plot()
#uncomment to print the membership functions
#D.optimal.plot()
#uncomment to print the membership functions
#D.close.plot()

#set the ranges for the membership functions and resolution
LA = Domain("Level of Attention", 0, 1, res=0.001)
#tune this values according to the ones desired for the membership functions
LA.low = S(0.3,0.5)
LA.medium = triangular(0.3,0.7)
LA.high = R(0.5, 0.7)
#uncomment to print the membership functions
#LA.low.plot()
#uncomment to print the membership functions
#LA.medium.plot()
#uncomment to print the membership functions
#LA.high.plot()

#Definition of RULES
R1 = Rule({(HR.middle, VR.middle, D.optimal): LA.high})
R2 = Rule({(HR.middle, VR.middle, D.far): LA.low})
R3 = Rule({(HR.middle, VR.middle, D.close): LA.high})
R4 = Rule({(HR.middle, VR.up, D.close): LA.low})
R5 = Rule({(HR.middle, VR.up, D.optimal): LA.low})
R6 = Rule({(HR.middle, VR.up, D.far): LA.low})
R7 = Rule({(HR.middle, VR.down, D.close): LA.medium})
R8 = Rule({(HR.middle, VR.down, D.optimal): LA.medium})
R9 = Rule({(HR.middle, VR.down, D.far): LA.low})  
R10 = Rule({(HR.left, VR.middle, D.optimal): LA.low})
R11 = Rule({(HR.left, VR.middle, D.far): LA.low})
R12 = Rule({(HR.left, VR.middle, D.close): LA.low})
R13 = Rule({(HR.left, VR.up, D.close): LA.low})
R14 = Rule({(HR.left, VR.up, D.optimal): LA.low})
R15 = Rule({(HR.left, VR.up, D.far): LA.low})
R16 = Rule({(HR.left, VR.down, D.close): LA.low})
R17 = Rule({(HR.left, VR.down, D.optimal): LA.low})
R18 = Rule({(HR.left, VR.down, D.far): LA.low})           
R19 = Rule({(HR.right, VR.middle, D.optimal): LA.low})
R20 = Rule({(HR.right, VR.middle, D.far): LA.low})
R21 = Rule({(HR.right, VR.middle, D.close): LA.low})
R22 = Rule({(HR.right, VR.up, D.close): LA.low})
R23 = Rule({(HR.right, VR.up, D.optimal): LA.low})
R24 = Rule({(HR.right, VR.up, D.far): LA.low})
R25 = Rule({(HR.right, VR.down, D.close): LA.low})
R26 = Rule({(HR.right, VR.down, D.optimal): LA.low})
R27 = Rule({(HR.right, VR.down, D.far): LA.low})

#Combination of rules  
rules = Rule({(HR.middle, VR.middle, D.optimal): LA.high,
              (HR.middle, VR.middle, D.far): LA.low, 
              (HR.middle, VR.middle, D.close): LA.high,
              (HR.middle, VR.up, D.close): LA.low,
              (HR.middle, VR.up, D.optimal): LA.low,
              (HR.middle, VR.up, D.far): LA.low,
              (HR.middle, VR.down, D.close): LA.medium,
              (HR.middle, VR.down, D.optimal): LA.medium,
              (HR.middle, VR.down, D.far): LA.low,  
              (HR.left, VR.middle, D.optimal): LA.low,
              (HR.left, VR.middle, D.far): LA.low,
              (HR.left, VR.middle, D.close): LA.low,
              (HR.left, VR.up, D.close): LA.low,
              (HR.left, VR.up, D.optimal): LA.low,
              (HR.left, VR.up, D.far): LA.low,
              (HR.left, VR.down, D.close): LA.low,
              (HR.left, VR.down, D.optimal): LA.low,
              (HR.left, VR.down, D.far): LA.low,           
              (HR.right, VR.middle, D.optimal): LA.low,
              (HR.right, VR.middle, D.far): LA.low,
              (HR.right, VR.middle, D.close): LA.low,
              (HR.right, VR.up, D.close): LA.low,
              (HR.right, VR.up, D.optimal): LA.low,
              (HR.right, VR.up, D.far): LA.low,
              (HR.right, VR.down, D.close): LA.low,
              (HR.right, VR.down, D.optimal): LA.low,
              (HR.right, VR.down, D.far): LA.low
             })

#Comparation of values of LA returned by each rule        
rules == R1 | R2 | R3 | R4 | R5 | R6 | R7 | R8 | R9 | R10 | R11 | R12 | R13 | R14 | R15 | R16 | R17 | R18 | R19 | R20 | R21 | R22 | R23 | R24 | R25 | R26 | R27 == sum([R1, R2, R3, R4, R5,R6,R7,R8,R9,R10,R11,R12,R13,R14,R15,R16,R17,R18,R18,R19,R20,R21,R22,R23,R24,R25,R26,R27])

#END FUZZY
#initialization values
Hr = 0.6
Vr = -0.6
Depth = -1030.5

print(__doc__)
print("OpenCV version: {}".format(cv2.__version__))

# Parse arguments from user input.
parser = ArgumentParser()
parser.add_argument("--video", type=str, default=None,
                    help="Video file to be processed.")
parser.add_argument("--cam", type=int, default=None,
                    help="The webcam index.")
args = parser.parse_args()





if __name__ == '__main__':
    # Before estimation started, there are some startup works to do.

    # 1. Setup the video source from webcam or video file.
    video_src = args.cam if args.cam is not None else args.video
    if video_src is None:
        print("Video source not assigned, default webcam will be used.")
        video_src = 0    #0 integrated webcam, 1 for usb webcam
    cap = cv2.VideoCapture(video_src)

    # Get the frame size. This will be used by the pose estimator.
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)

    # 2. Introduce a pose estimator to solve pose.
    pose_estimator = PoseEstimator(img_size=(height, width))

    # 3. Introduce a mark detector to detect landmarks.
    mark_detector = MarkDetector()

    # 4. Measure the performance with a tick meter.
    tm = cv2.TickMeter()

    # Now, let the frames flow.
    while True:

        # Read a frame.
        frame_got, frame = cap.read()
        if frame_got is False:
            break

        # If the frame comes from webcam, flip it so it looks like a mirror.
        if video_src == 1: 
            frame = cv2.flip(frame, 2)

        # Step 1: Get a face from current frame.
        facebox = mark_detector.extract_cnn_facebox(frame)

        # Any face found?
        if facebox is None: ##################
            print('No face detected: ')#######
            #if there is no face detected the values for the inputs will be 
            Hr = -1.5
            Vr = -0.85
            Depth = -1300
            # making the fuzzy Level of attention output a low attention
            print('Horizontal Rotation',Hr)
            print('Vertical Rotation',Vr)
            print('Depth',Depth)
            
            #save values on .csv files
            f = open('HorRotation.csv','a')
            try:    
             f.write(str(Hr).split('[')[1].split(']')[0])
            except:
             f.write(str(Hr))	
            f.write('\n')
            f.close()

            f = open('VerRotation.csv','a')
            try:
             f.write(str(Vr).split('[')[1].split(']')[0])
            except:
             f.write(str(Vr))
            f.write('\n')
            f.close()

            f = open('Depth.csv','a')
            try:
             f.write(str(Depth).split('[')[1].split(']')[0])
            except:
             f.write(str(Depth))
            f.write('\n')
            f.close()

        if facebox is not None:

            # Step 2: Detect landmarks. Crop and feed the face area into the
            # mark detector.
            x1, y1, x2, y2 = facebox
            face_img = frame[y1: y2, x1: x2]

            # Run the detection.
            tm.start()
            marks = mark_detector.detect_marks(face_img)
            tm.stop()

            # Convert the locations from local face area to the global image.
            marks *= (x2 - x1)
            marks[:, 0] += x1
            marks[:, 1] += y1

            # Try pose estimation with 68 points.
            pose = pose_estimator.solve_pose_by_68_points(marks)

            # All done. The best way to show the result would be drawing the
            # pose on the frame in realtime.

            # Do you want to see the pose annotation?
            pose_estimator.draw_annotation_box(
                frame, pose[0], pose[1], color=(0, 255, 0))


            Hr = pose[0][0]
            Vr = pose[0][1]
            Depth = pose[1][2]
            print('Horizontal Rotation',Hr)
            print('Vertical Rotation',Vr)
            print('Depth',Depth)
            #save values on .csv files
            f = open('HorRotation.csv','a')
            try:    
             f.write(str(Hr).split('[')[1].split(']')[0])
            except:
             f.write(str(Hr))
            f.write('\n')
            f.close()

            f = open('VerRotation.csv','a')
            try:
             f.write(str(Vr).split('[')[1].split(']')[0])
            except:
             f.write(str(Vr))
            f.write('\n')
            f.close()

            f = open('Depth.csv','a')
            try:
             f.write(str(Depth).split('[')[1].split(']')[0])
            except:
             f.write(str(Depth))
            f.write('\n')
            f.close()	
            # Do you want to see the head axes?
            # pose_estimator.draw_axes(frame, pose[0], pose[1])

            # Do you want to see the marks?
            # mark_detector.draw_marks(frame, marks, color=(0, 255, 0))

            # Do you want to see the facebox?
            # mark_detector.draw_box(frame, [facebox])

        # Show preview.
        

        values =  {HR: Hr, VR: Vr, D:Depth} 
#Robot needs to receive Level_of_Attention,        
        Level_of_Attention = rules (values)
        
        print('Level of Attention',rules(values))
        if(rules(values)<0.4):
            print("Level of Attention: LOW")
        if(rules(values)>=0.4 and rules(values)<0.6):
            print("Level of Attention: MEDIUM")     
        if(rules(values)>=0.6):
            print("Level of Attention: HIGH")
        #Delay for tuning
        time.sleep(1.1) #add a delay so you can see the value while you are looking at specific points in the screen
        #Display video window       
        cv2.imshow("Pose", frame)
        if cv2.waitKey(1) == 27:
            break
