# Face_Det_Level_of_Attention
Detect if the user is focused on the screen by measuring the head's vertical and horizontal rotations and the distance from the screen and passing these three into a fuzzy logic algorithm to get a value of the level of attention.

For this code we used:
The fuzzy library from https://github.com/amogorkon/fuzzylogic
As well as the face detection algorithm from https://github.com/yinguobing/head-pose-estimation.git
Please follow the instructions in the respective repositories to set up everything.
Once everything is installed you can run the code.

Running
python Level_Attention.py

For exiting the program press esc while viewing the video window.

Note
The level of attention values have to be tuned manually according to the webcam used and the operator.
