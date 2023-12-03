import cv2
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage
import scipy.signal as sig
from scipy import misc
import imageio

def upsample(image, dst_shape):
  return cv2.pyrUp(image, dstsize=(dst_shape[1], dst_shape[0]))

def downsample(image):    
  cols = image.shape[1] // 2
  rows = image.shape[0] // 2                                             
  return cv2.pyrDown(image, dstsize=(cols, rows))    
                     
def pyramids(image):
  gaussian_pyramid = [image, ]
  laplacian_pyramid = []

  while image.shape[0] > 2 and image.shape[1] > 2:
    image = downsample(image)
    gaussian_pyramid.append(image)

  for i in range(len(gaussian_pyramid) - 1):
    laplacian_pyramid.append(gaussian_pyramid[i] - upsample(gaussian_pyramid[i + 1], gaussian_pyramid[i].shape))

  return gaussian_pyramid[:-1], laplacian_pyramid

def pyramidBlending(A, B, mask):
  _, source_LA = pyramids(A)
  _ ,dest_LB = pyramids(B)
  Gmask, _ = pyramids(mask)
  blend = []
  for i in range(len(source_LA)):
    LS = (Gmask[i]/255)*source_LA[i] + (1-(Gmask[i]/255))*dest_LB[i]
    blend.append(LS)
  return blend

def collapse_pyramid(pyramid):
  small_to_big_pyramid = pyramid[::-1]
  collapsed_output = small_to_big_pyramid[0]
  for i in range(1, len(small_to_big_pyramid)):
    collapsed_output = upsample(collapsed_output, small_to_big_pyramid[i].shape) + small_to_big_pyramid[i] # upsampling simultaneously
  return collapsed_output

def colorBlending(img1, img2, mask):
  img1R,img1G,img1B = cv2.split(img1)
  img2R,img2G,img2B = cv2.split(img2)
  R = collapse_pyramid(pyramidBlending(img1R, img2R, mask))
  G = collapse_pyramid(pyramidBlending(img1G, img2G, mask))
  B = collapse_pyramid(pyramidBlending(img1B, img2B, mask))
  output = (cv2.merge((B, G, R)))
  output = 0 + (255/(np.max(output) - np.min(output))) * (output - np.min(output))
  return output

output_file_name = "3_try_output.jpg"
destination = cv2.cvtColor(cv2.imread('3_try_piyush.jpg'), cv2.COLOR_BGR2RGB).astype(np.float32)
source = cv2.cvtColor(cv2.imread('3_try_harsh.jpg'), cv2.COLOR_BGR2RGB).astype(np.float32)
mask_from_source = cv2.cvtColor(cv2.imread('mask_3_try_harsh.jpg'), cv2.COLOR_BGR2GRAY).astype(np.float32)
output = colorBlending(source, destination, mask_from_source)
cv2.imwrite(output_file_name, output)
