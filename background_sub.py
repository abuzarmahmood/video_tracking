import cv2
import os
import numpy as np
from tqdm import tqdm
import pylab as plt


data_path = '/media/bigdata/projects/video_tracking/data/GW15_OFday1.mp4'

resize_shape = (640, 480)
crop_corners = [
        (235, 75),
        (226, 446),
        (501, 451),
        (506, 89)
        ]

min_x, max_x = min(crop_corners[0][0], crop_corners[1][0]), max(crop_corners[2][0], crop_corners[3][0])
min_y, max_y = min(crop_corners[0][1], crop_corners[3][1]), max(crop_corners[1][1], crop_corners[2][1])

cap = cv2.VideoCapture(data_path)
# Check if camera opened successfully
if (cap.isOpened()== False): 
  print("Error opening video stream or file")
 
# Read until video is completed
while(cap.isOpened()):
  # Capture frame-by-frame
  ret, frame = cap.read()
  if ret == True:
 
    # Display the resulting frame
    resize_frame = cv2.resize(frame, resize_shape) 
    crop_frame = resize_frame[min_y:max_y, min_x:max_x]
    cv2.imshow('Frame', crop_frame)
 
    # Press Q on keyboard to  exit
    if cv2.waitKey(25) & 0xFF == ord('q'):
      break
 
  # Break the loop
  else: 
    break
 
# When everything done, release the video capture object
cap.release()
 
# Closes all the frames
cv2.destroyAllWindows()

############################################################
# Write out cropped video 
############################################################
out_dir = os.path.dirname(data_path)
basename = os.path.basename(data_path)
out_path = os.path.join(out_dir, 'cropped_' + basename)

crop_video_shape = (max_x - min_x, max_y - min_y)

cap = cv2.VideoCapture(data_path)

# Get fps
fps = int(np.round(cap.get(cv2.CAP_PROP_FPS),0))

# Get total number of frames
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

out = cv2.VideoWriter(out_path,
                      cv2.VideoWriter_fourcc('M','J','P','G'), 
                      fps, 
                      crop_video_shape)

# Check if camera opened successfully
if (cap.isOpened()== False): 
  print("Error opening video stream or file")
 
# Read until video is completed
# Create tqdm progress bar
pbar = tqdm(total=total_frames)

while(cap.isOpened()):
  # Capture frame-by-frame
  ret, frame = cap.read()
  pbar.update(1)
  if ret == True:
 
    # Display the resulting frame
    resize_frame = cv2.resize(frame, resize_shape)
    crop_frame = resize_frame[min_y:max_y, min_x:max_x]
    out.write(crop_frame)
 
    # Press Q on keyboard to  exit
    if cv2.waitKey(25) & 0xFF == ord('q'):
      break

  # Break the loop
  else: 
    break
 
# When everything done, release the video capture object
cap.release()
out.release()
 
############################################################
# Perform background subtraction
############################################################
backSub = cv2.createBackgroundSubtractorMOG2()

cap = cv2.VideoCapture(out_path)
# Check if camera opened successfully
if (cap.isOpened()== False): 
  print("Error opening video stream or file")
 
# Read until video is completed
# Create tqdm progress bar
pbar = tqdm(total=total_frames)

while(cap.isOpened()):
  # Capture frame-by-frame
  ret, frame = cap.read()
  pbar.update(1)
  if ret == True:
 
    fgMask = backSub.apply(frame)
    cv2.imshow('Frame', fgMask)
 
    # Press Q on keyboard to  exit
    if cv2.waitKey(25) & 0xFF == ord('q'):
      break

  # Break the loop
  else: 
    break
 
# When everything done, release the video capture object
cap.release()

# Closes all the frames
cv2.destroyAllWindows()
