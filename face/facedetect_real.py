import cv2
import numpy as np

#The part in dash line is my first Histogram Equalization algorithm, but the delay is high
"""
#----------------------------------------------------------------------------
# Function imhist calculates normalized histogram of an image
def imhist(im):

    size=im.shape[:2]
    m=size[0]
    n=size[1]
    h=[0.0]*256

    for i in range(m):
        for j in range(n):
            h[im[i,j]]+=1

    return np.array(h)/(m*n)

 # Function cumsum finds cumulative sum of a numpy array, list
def cumsum(h):
    return [sum(h[:i+1]) for i in range(len(h))]

# Function histeq calculates Histogram
def histeq(im):

    h=imhist(im)
    cdf=np.array(cumsum(h)) #cumulative distribution function

    sk=np.uint8(255 * cdf) #finding transfer function values

    s1,s2=im.shape
    Y=np.zeros_like(im)
    # applying transfered values for each pixels

    for i in range(0, s1):
        for j in range(0, s2):
            Y[i,j]=sk[im[i,j]]
    H=imhist(Y)

    return Y
"""
#----------------------------------------------------------------------------------------------------------

cv2.namedWindow("face_detect") 
cap=cv2.VideoCapture(0) 

print cap.isOpened() # test if the camera is on
success,frame=cap.read()

findface=cv2.CascadeClassifier("haarcascade_frontalface_alt.xml") 
findeyes=cv2.CascadeClassifier('haarcascade_eye.xml')


while success:

    success,frame=cap.read()
    color1=(255,255,255) 
    color2=(0,255,0)

    size=frame.shape[:2] # get the size of image
    image=cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) # convert the image to RGB

    #image = histeq(image) # My first Histogram Equalization algorithm, but delay is high

    hist,bins=np.histogram(image.flatten(),256,[0,256])
    cdf=hist.cumsum()
    cdf_m=np.ma.masked_equal(cdf,0)
    cdf_m=(cdf_m - cdf_m.min())*255/(cdf_m.max()-cdf_m.min())
    cdf=np.ma.filled(cdf_m,0).astype('uint8')

    image=cdf[image]
    
    divisor=20
    h,w=size
    minSize=(w/divisor, h/divisor)# min size of image

    face=findface.detectMultiScale(image, 1.2, 3, cv2.CASCADE_SCALE_IMAGE,minSize) 

    if len(face)>0:
        for (x,y,w,h) in face:  #Find out every face and set rectangle parameters
                cv2.rectangle(frame, (x, y), (x+w, y+h), color1, 2) # Draw rectangles on faces
                eye_gray=image[y:y+h, x:x+w]
                eye_color=frame[y:y+h, x:x+w]
        
                eyes=findeyes.detectMultiScale(eye_gray)
                for (ex,ey,ew,eh) in eyes:
                    cv2.rectangle(eye_color, (ex,ey), (ex+ew,ey+eh), color2, 2) # Draw rectangles on eyes

    cv2.imshow("test", frame)

    key=cv2.waitKey(10) #Display image for 10 ms

    c=chr(key & 255)
    if c in ['q', 'Q', chr(27)]:
        break

    # If user want to quit, press q
cv2.destroyWindow("test")