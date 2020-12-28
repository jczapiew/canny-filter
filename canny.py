from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

image = Image.open('lena.png')
width, height = image.size

fig, ((ax1, ax2, ax3), (ax4, ax5, ax6)) = plt.subplots(2, 3)

ax2.imshow(image)
#gray histogram

image = np.asarray(image, dtype='int32')

red = image[:, :, 0]
green = image[:, :, 1]
bleu = image[:, :, 2]

gray = np.add(0.299*image[:, :, 0], 0.587*image[:, :, 1], 0.114*image[:, :, 2])
gray1 = gray.astype(int) // 4

histogram = np.zeros(64, dtype='int32')
items, count = np.unique(gray1, return_counts=True)
histogram[items] = count
x_axes = np.arange(64)
ax1.bar(x_axes,histogram,color='gray')
ax3.imshow(Image.fromarray(gray))

#gaussian blur

sigma = 60
x = np.linspace(0, width, width)
y = np.linspace(0, height, height)
X, Y = np.meshgrid(x, y)
gaussian_blur = np.exp(-((((X-(width/2))/sigma)**2)/2 + (((Y-(height/2))/sigma)**2)/2))

out_img = gray
ftout_img = np.fft.fft2(out_img)
ftout_img = np.fft.fftshift(ftout_img)

ftout_imgr = ftout_img * gaussian_blur

out_imgr = np.fft.ifft2(ftout_imgr)
out_img = Image.fromarray(np.abs(out_imgr))
ax4.imshow(out_img)

#intensitivity gradient
out_imgr00 = np.zeros((out_imgr.shape[0]+1,out_imgr.shape[1]+1), dtype='int32')
out_imgr11 = np.zeros((out_imgr.shape[0]+1,out_imgr.shape[1]+1), dtype='int32')
out_imgr10 = np.zeros((out_imgr.shape[0]+1,out_imgr.shape[1]+1), dtype='int32')
out_imgr01 = np.zeros((out_imgr.shape[0]+1,out_imgr.shape[1]+1), dtype='int32')

out_imgr00[:-1,:-1] = np.abs(out_imgr)
out_imgr11[1:,1:] = np.abs(out_imgr)
out_imgr10[1:,:-1] = np.abs(out_imgr)
out_imgr01[:-1,1:] = np.abs(out_imgr)

Gx = out_imgr00 - out_imgr11
Gy = out_imgr10 - out_imgr01

G = np.sqrt(np.square(Gx) + np.square(Gy))

out_edge = Image.fromarray(G)
ax5.imshow(out_edge)

gradient = 180*np.arctan2(Gx,Gy)/np.pi

gradient_pusty = np.zeros(gradient.shape[0]*gradient.shape[1]).reshape(gradient.shape[0], gradient.shape[1])
gradient_pusty = np.where(np.logical_or(np.logical_and(gradient >= 22.5, gradient < 67.5), np.logical_and(gradient < -112.5, gradient >= -157.5)), 45, gradient_pusty)
gradient_pusty = np.where(np.logical_or(np.logical_and(gradient >= 67.5, gradient < 112.5), np.logical_and(gradient < -67.5, gradient >= -112.5)), 90, gradient_pusty)
gradient_pusty = np.where(np.logical_or(np.logical_and(gradient >= 112.5, gradient < 157.5), np.logical_and(gradient < -22.5, gradient >= -67.5)), 135, gradient_pusty)

#Non-maximum suppression

nms_grad = np.zeros(G.shape[0]*G.shape[1]).reshape(G.shape[0],G.shape[1])
nms_grad = np.where(np.logical_and(gradient_pusty == 0, G > np.roll(G,1,axis=1), G > np.roll(G,-1,axis=1)), 255, nms_grad)
nms_grad = np.where(np.logical_and(gradient_pusty == 90, G > np.roll(G,1,axis=0), G > np.roll(G,-1,axis=0)), 255, nms_grad)
nms_grad = np.where(np.logical_and(gradient_pusty == 45, G > np.roll(G,[1,1],axis=(0,1)), G > np.roll(G,[-1, -1], axis=(0, 1))), 255, nms_grad)
nms_grad = np.where(np.logical_and(gradient_pusty == 135, G > np.roll(G,[1, -1], axis=(0, 1)), G > np.roll(G,[-1, 1], axis=(0, 1))), 255, nms_grad)

#treshold
border = 17
nms_grad = np.where( G > border, 255, 0)

ax6.imshow(Image.fromarray(nms_grad))

plt.show()