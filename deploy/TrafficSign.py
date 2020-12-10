
#################### import all libraries and initializations ############

import sys
import numpy as np 
import os
import time
import math
from PIL import Image
import cv2
from datetime import datetime
from pynq import Xlnk
from pynq import Overlay
import pynq
import struct
from multiprocessing import Process, Pipe, Queue, Event, Manager

print('\n**** Running SkyNet')

xlnk = Xlnk()
xlnk.xlnk_reset()


########## Allocate memory for weights and off-chip buffers
mytype = 'B,'*31 + 'B'
dt = np.dtype(mytype)
img = xlnk.cma_array(shape=(3,58,58), dtype=np.uint8)
conv_weight_1x1_all = xlnk.cma_array(shape=(59,16*32), dtype=np.int8)
conv_weight_3x3_all = xlnk.cma_array(shape=(12,9*32), dtype=np.int8)
bias_all = xlnk.cma_array(shape=(31*32), dtype=np.int8)
DDR_dw1_pool_out = xlnk.cma_array(shape=(900*32), dtype=np.int8)
DDR_dw2_pool_out = xlnk.cma_array(shape=(512*32), dtype=np.int8)
DDR_buf = xlnk.cma_array(shape=(1620*32), dtype=np.int8)

cla = xlnk.cma_array(shape=(1), dtype=np.int)

print("Allocating memory done")


########### Load parameters from SD card to DDR
params = np.fromfile("TrafficSign.bin", dtype=np.int8)
idx = 0
np.copyto(conv_weight_1x1_all, params[idx:idx+conv_weight_1x1_all.size].reshape(conv_weight_1x1_all.shape))
idx += conv_weight_1x1_all.size
np.copyto(conv_weight_3x3_all, params[idx:idx+conv_weight_3x3_all.size].reshape(conv_weight_3x3_all.shape))
idx += conv_weight_3x3_all.size
np.copyto(bias_all, params[idx:idx+bias_all.size].reshape(bias_all.shape))


################### Download the overlay
overlay = Overlay("./TrafficSign.bit")
print("Bitstream loaded")



################## Utility functions 

# IMG_DIR = '/home/xilinx/jupyter_notebooks/TrafficSign/'
# # Get image name list
# def get_image_names():
#     names_temp = [f for f in os.listdir(IMG_DIR) if f.endswith('.jpg')]
#     names_temp.sort(key= lambda x:int(x[:-4]))
#     return names_temp

# # Process the images in batches, may help when write to XML
# BATCH_SIZE = 4
# def get_image_batch():
#     image_list = get_image_names()
#     batches = list()
#     for i in range(0, len(image_list), BATCH_SIZE):
#         batches.append(image_list[i:i+BATCH_SIZE])
#     return batches

# def stitch(image_queue, name_queue):
#     blank = Image.new('RGB', (644, 324), (127, 127, 127))
#     img = np.ndarray(shape=(3,162*2,322*2), dtype=np.uint8)
    
#     for batch in get_image_batch():
#         for i in range(0, len(batch), 4):
#             while image_queue.full():
#                 continue
            
#             pic_name = IMG_DIR + batch[0]
#             image = Image.open(pic_name).convert('RGB')
#             image = image.resize((320, 160))
#             blank.paste(image, (1, 1))

#             pic_name = IMG_DIR + batch[1]
#             image = Image.open(pic_name).convert('RGB')
#             image = image.resize((320, 160))
#             blank.paste(image, (323, 1))

#             pic_name = IMG_DIR + batch[2]
#             image = Image.open(pic_name).convert('RGB')
#             image = image.resize((320, 160))
#             blank.paste(image, (1, 163))

#             pic_name = IMG_DIR + batch[3]
#             image = Image.open(pic_name).convert('RGB')
#             image = image.resize((320, 160))
#             blank.paste(image, (323, 163))

#             image_stitched = np.transpose(blank, (2, 0, 1))
#             image_queue.put(image_stitched)
            
            
# def compute_bounding_box(boxes, output_queue):
#     predict_boxes = np.empty([4, 5], dtype=np.float32)
#     constant = np.empty([4, 3], dtype=np.int32)
    
#     for batch in get_image_batch():
#         #print(batch)
#         for i in range(0, len(batch), 4):
            
#             while output_queue.empty():
#                 continue
                
#             outputs = output_queue.get()
#             outputs_boxes = outputs[0]
#             outputs_index = outputs[1]
#             np.copyto(predict_boxes, np.array(outputs_boxes))
#             np.copyto(constant, np.array(outputs_index))
                
#             for idx in range(0, 4):
#                 predict_boxes[idx][0] = 1.0 / (1.0 + math.exp(-predict_boxes[idx][0])) + constant[idx][1];
#                 predict_boxes[idx][1] = 1.0 / (1.0 + math.exp(-predict_boxes[idx][1])) + constant[idx][2];

#                 if( constant[idx][0] == 0 ):
#                     predict_boxes[idx][2] = math.exp(predict_boxes[idx][2]) * box[0];
#                     predict_boxes[idx][3] = math.exp(predict_boxes[idx][3]) * box[1];
#                 else:
#                     predict_boxes[idx][2] = math.exp(predict_boxes[idx][2]) * box[2];
#                     predict_boxes[idx][3] = math.exp(predict_boxes[idx][3]) * box[3];
#                 predict_boxes[idx][4] = 1.0 / (1.0 + math.exp(-predict_boxes[idx][4]));

#                 predict_boxes[idx][0] = predict_boxes[idx][0] / 40;
#                 predict_boxes[idx][1] = predict_boxes[idx][1] / 20;
#                 predict_boxes[idx][2] = predict_boxes[idx][2] / 40;
#                 predict_boxes[idx][3] = predict_boxes[idx][3] / 20;
#                 #print(predict_boxes[idx])

#                 x1 = int(round((predict_boxes[idx][0] - predict_boxes[idx][2]/2.0) * 640))
#                 y1 = int(round((predict_boxes[idx][1] - predict_boxes[idx][3]/2.0) * 360))
#                 x2 = int(round((predict_boxes[idx][0] + predict_boxes[idx][2]/2.0) * 640))
#                 y2 = int(round((predict_boxes[idx][1] + predict_boxes[idx][3]/2.0) * 360))
#                 result_rectangle.append([x1, x2, y1, y2])

#                 #print([x1, x2, y1, y2])


###########################################################
################ MAIN PART OF DETECTION ###################
###########################################################

SkyNet = overlay.SEUer_0

SkyNet.write(0x20, img.physical_address)
SkyNet.write(0x30, conv_weight_1x1_all.physical_address)
SkyNet.write(0x40, conv_weight_3x3_all.physical_address)
SkyNet.write(0x50, bias_all.physical_address)
SkyNet.write(0x60, DDR_dw1_pool_out.physical_address)
SkyNet.write(0x70, DDR_dw2_pool_out.physical_address)
SkyNet.write(0x80, DDR_buf.physical_address)
SkyNet.write(0x90, cla.physical_address)


#rails = pynq.get_rails()
#recorder = pynq.DataRecorder(rails['5V'].power)

################# Declare New Process ##############
# image_queue = Queue(200) ## could be smaller
# name_queue = Queue(200)
# output_queue = Queue(10)
# mgr = Manager()
# result_rectangle = mgr.list()
# p1 = Process(target=stitch, args=(image_queue, name_queue))
# p2 = Process(target=compute_bounding_box, args=(result_rectangle, output_queue))

################### Start to detect ################
# output_boxes = np.empty([4, 5], dtype=np.float32)
# output_index = np.empty([4, 3], dtype=np.int32)

# p1.start()
# p2.start()
print("\n**** Start to detect")
start = time.time()

# for batch in get_image_batch():
#     for i in range(0, len(batch), 4):

#         while image_queue.empty():
#             continue

#         preprocessed_img = image_queue.get()
#         np.copyto(img, np.array(preprocessed_img))

#         SkyNet.write(0x00, 1)
#         isready = SkyNet.read(0x00)
#         while( isready == 1 ):
#             isready = SkyNet.read(0x00)

#         outputs = []
#         np.copyto(output_boxes, predict_boxes)
#         np.copyto(output_index, constant)
#         outputs.append(output_boxes)
#         outputs.append(output_index)
#         output_queue.put(outputs)
# p1.join()   
# p2.join()
# last_cla=-1
for root, dirs, files in os.walk('./traffic-sign/test/00021'):
    for file in files:
#         print(file)
        #讀入圖像
        img_path = root+'/'+file
        preprocessed_img = Image.open(img_path).convert('RGB')
#         print("Image open")
        preprocessed_img = preprocessed_img.resize((56, 56))
#         print("Image resize")
        blank = Image.new('RGB', (58, 58), (127, 127, 127))
        blank.paste(preprocessed_img, (1, 1))
        image_stitched = np.transpose(blank, (2, 0, 1))
        np.copyto(img, image_stitched)
#         print("Image loaded")

        SkyNet.write(0x00, 1)
        isready = SkyNet.read(0x00)
        while( isready == 1 ):
            isready = SkyNet.read(0x00)        
        print(cla[0])
        
#         last_cla=cla[0]
        
        
    
    
print("**** Detection finished\n")
        
end = time.time()
total_time = end - start
print('Total time: ' + str(total_time) + ' s')

#energy = recorder.frame["5V_power"].mean() * total_time
#print('Total energy: ' + str(energy) + ' J')

############## clean up #############
xlnk.xlnk_reset()  

