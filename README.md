# FPGA-based-Traffic-Sign-Recognition
We designed a traffic sign recognition system that can recognize traffic signs quickly. 

## Team
Yu Zhang </br>
Xinye Cao 

## Things
Ultra96 v2  *1

## Story
Nowadays, the traffic network is very developed, and the traffic signs are the facilities that use graphic symbols and words to convey specific information to manage the traffic and indicate the direction of driving to ensure the smooth and safe operation of the road. Traffic signs are divided into main signs and auxiliary signs. The main signs are divided into warning signs, prohibition signs, indication signs, guide signs, tourist area signs and road construction safety signs. There are many kinds of traffic signs and their graphics are similar, so novice drivers often face great difficulties in learning. Moreover, modern urban roads crisscross each other. If the driver misreads the traffic signs, they may take the wrong road, run the red light, go retrograde, violate the stop and so on, and even may cause traffic accidents.
Under the above background, this project designs a traffic sign recognition system, which can quickly identify the traffic signs in the image according to the input image. The system can be integrated into the vehicle assistant driving system. According to the current road image information, the system can calculate the traffic prompt information contained in the front sign in real time, and feed back to the driver and other systems in time, so as to alleviate the driver's labor intensity and protect the driver's safety. This design can also be applied to intelligent transportation system to realize the interaction and coordination of people, vehicles and roads. It plays an important role in regulating traffic behavior, indicating road conditions, guiding pedestrians and driving safely, and can reduce traffic congestion and accidents to a certain extent. In addition, the traffic sign recognition system is also an important part of the automatic driving system. The traffic sign recognition system provides timely road information for vehicles by identifying traffic signs on the road in real time, and helps the automatic driving vehicle to choose the right road.
The traffic sign recognition system designed in this project has the characteristics of high accuracy and high real-time, which can meet the requirements of accuracy and real-time in most application scenarios. This design uses the lightweight neural network based on Skynet network structure to learn the traffic sign recognition task. There are 62 kinds of traffic signs in the data set, including 4572 photos in the training set (about 70 in each category) and 2520 photos (about 40 in each category) in the test data set. The accuracy of the trained network is 99% on the training set and 93.5% on the test set. In this project, the trained neural network is deployed on FPGA (pynq-z2 board) to obtain high real-time processing ability. In the hardware design, high level synthesis (HLS) is used to write the functional logic of the hardware circuit to realize the neural network structure, and the optimization instructions are added in the appropriate position to improve the parallelism of the circuit, so as to reduce the delay and make the system achieve high real-time performance.

## Attachments


