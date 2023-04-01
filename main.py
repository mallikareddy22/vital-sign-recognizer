import cv2
import easyocr
import pytesseract
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont

image_path = 'vitals1.png'

def preprocess(read_image, resize=False):
    #ret, imgf = cv2.threshold(
        #read_image, 110, 255,cv2.THRESH_BINARY,cv2.THRESH_OTSU)
    #if resize:
        #resize_img = cv2.resize(imgf, None, fx=1.2, fy=1.4, interpolation=cv2.INTER_CUBIC)
    #else:
        #resize_img = imgf
    #resize_img = cv2.resize(read_image, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
    grayscale_resize_img = cv2.cvtColor(
        read_image, cv2.COLOR_BGR2GRAY)
    gaussian_blur_img = cv2.GaussianBlur(
        grayscale_resize_img, (5, 5), 0)
    alpha = 1.5 # Contrast control (1.0-3.0)
    beta = 30 # Brightness control (0-100)
    adjusted = cv2.convertScaleAbs(gaussian_blur_img, alpha=alpha, beta=beta)
    return adjusted

     
img = cv2.imread(image_path)
new_image = preprocess(img)

#kernel = np.ones((5,5),np.uint8)
#erosion = cv2.erode(gaussian_blur_img,kernel,iterations = 1)

#plt.imshow(adjusted, cmap="gray")
#plt.show()

#create boxes for the number and the word

reader = easyocr.Reader(['en'])


#im = Image.open(image_path)
results = reader.readtext(new_image)

low_precision = []
for text in results:
    if text[2]<0.40: # precision here
        low_precision.append(text)
for i in low_precision:
    results.remove(i) # remove low precision
#print(results)

for text in results:
    print(text[1])
#print(text)

print(results)
vitals = {}

keys = []
values = []
all_val = []

for bound in results:
        adjusted_copy = img.copy()
        #move left top (p3) and left bottom (p0) x coordinate[0] left by 500
        #move left top (p3) and right top (p2) y coordinate[1] up by 100
        #p0 left bottom
        #p1 right bottom
        #p2 right top
        #p3 left top
        p0, p1, p2, p3 = bound[0]
        print(p0, p1, p2, p3)
        p3[0] = p3[0] - 30
        p1[1] = p1[1] - 15
        #cropped = img[start_row:end_row, start_col:end_col]
        adjusted_copy = adjusted_copy[p1[1]:p3[1], p3[0]:p1[0]]
        print(p0, p1, p2, p3)
        
        if(adjusted_copy is not None):
            #cropped_new_image = preprocess(adjusted_copy)
            cropped_new_image = adjusted_copy
            cropped_new_image = cv2.resize(adjusted_copy, None, fx=5, fy=5, interpolation=cv2.INTER_CUBIC)
            plt.imshow(cropped_new_image, cmap="gray")
            plt.show()
            new_result = reader.readtext(cropped_new_image)

            for i in new_result:
                print(i[1])
                if not ((i[1].isdigit()) or ('(' in i[1]) or ('/' in i[1]) or ('.' in i[1]) or ('-' in i[1]) or ('$' in i[1])):
                    if (len(i[1])>2 or i[1] == 'HR'):
                        vitals[i[1].upper()] = bound[1]
                        break

            '''
            for i in new_result:
                print(i[1])
                all_val.append(i[1])
            if len(all_val) > 0:
                if len(all_val) > 1:
                    keys.append(all_val[0])
                    values.append(all_val[-1])
                else:
                    values.append(all_val[0])
            all_val = []
            '''

                
for (coord, text, prob) in results:

  (topleft, topright, bottomright, bottomleft) = coord
  tx,ty = (int(topleft[0]), int(topleft[1]))
  bx,by = (int(bottomright[0]), int(bottomright[1]))
  cv2.rectangle(img, (tx,ty), (bx,by), (0, 0, 255), 2)

plt.imshow(img)
plt.show()


print(vitals)


'''
def draw_boxes(image, bounds, color='yellow', width=2):
    draw = ImageDraw.Draw(image)
    for bound in bounds:
        p0, p1, p2, p3 = bound[0]
        draw.line([*p0, *p1, *p2, *p3, *p0], fill=color, width=width)
    return image

draw_boxes(im, text).show()
'''

