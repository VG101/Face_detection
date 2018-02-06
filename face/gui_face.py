from Tkinter import *
import os

root = Tk(className = 'Face_detection_and_recognition')
username = StringVar()

w = Entry(root,textvariable=username) 
w.pack()
def detect():
    os.system('python facedetect_real.py')

def train():
    name = username.get()
    os.system('python face_train_eigen.py %s'%name)

def recog():
    os.system('python face_recog_eigen.py')

trainE_btn = Button(root,text="Detect", command=detect)
trainE_btn.pack()

trainE_btn = Button(root,text="Train (EigenFaces)", command=train)
trainE_btn.pack()

recogE_btn = Button(root,text="Recognize (EigenFaces)", command=recog)
recogE_btn.pack()

root.mainloop()