import cv2
import numpy as np
from os import listdir
from os.path import isfile, join
face_classifier = cv2.CascadeClassifier('D:/Learning/python_openCV/haarcascade_frontalface_default.xml')


def train():
    data_path = 'D:/Learning/python_openCV/faces/'
    only_files = [f for f in listdir(data_path) if isfile(join(data_path, f))]

    training_data, labels = [], []
    for i, files in enumerate(only_files):
        image_path = data_path + only_files[i]
        images = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        training_data.append(np.asarray(images, dtype=np.uint8))
        labels.append(i)
    labels = np.asarray(labels, dtype=np.int32)    
    model = cv2.face.LBPHFaceRecognizer_create()
    model.train(np.asarray(training_data), np.asarray(labels))
    model.save ('D:/Learning/python_openCV/faces/trained.yml')
    print("done learing faces")

def face_extractor(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray, 1.3, 5)
    if faces is():
        return None
    for (x, y, w, h) in faces:
        cropped_faces = img[y:y+h, x:x+w]
    return cropped_faces    

def take_samples():   
    cap = cv2.VideoCapture(0)
    count = 0
    while True:
        _, frame = cap.read()
        if face_extractor(frame) is not None:
            count += 1
            face = cv2.resize(face_extractor(frame),(200, 200))
            face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
            file_name_path = 'D:/Learning/python_openCV/faces/user'+str(count)+'.jpg'
            cv2.imwrite(file_name_path,face)
            cv2.putText(face, str(count), (50,50), cv2.FONT_HERSHEY_COMPLEX, 1, (0,255,0), 2)
            cv2.imshow('face ', face)
        else:
            print('No Faces or too ugly for a computer')
            pass
        if cv2.waitKey(1) == 13 or count == 50:
            break
    cap.release()
    cv2.destroyAllWindows()
    print('got faces')

    
def face_dectector(img, size = 0.5):
    gray = gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray, 1.3, 5)
    if faces is():
        return img, []
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x+w, y+h), (0,255,255), 3)   
        roi = img[y:y+h, x:x+w]
        roi = cv2.resize(roi,(200,200))
        return img, roi

def match_face():
    cap = cv2.VideoCapture(0)
    while True:
        _,frame = cap.read()
        image, face = face_dectector(frame)
        try:
            model = 'D:/Learning/python_openCV/faces/trained.yml'
            face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
            result = model.predict(face)
            if result[1] > 500:
                confidence  = int(100 * (1 - (result[1]/300)))
                display = str(confidence) + ' %'
            cv2.putText(image, display, (100,120), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0))
            if confidence > 75:
                cv2.putText(image, 'found you', (100,120), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0))
                cv2.imshow('Face crop', image)
            else:
                cv2.putText(image, 'thief', (100,120), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255))
                cv2.imshow('Face crop', image)    
        except:
            cv2.putText(image, 'noFace', (100,120), cv2.FONT_HERSHEY_SIMPLEX, 2, (0,0,255))            
            cv2.imshow('Face crop', image)
            pass
        if cv2.waitKey(1) == 13:
            break
    cap.release()
    cv2.destroyAllWindows()        
input_user = input("what to do? 1: take photos 2: train ")
input_user = int(input_user)
if input_user == 1:
    print ("1 is selected")
    take_samples()
elif input_user ==2:
    print ("2 is selected")
    train()
elif input_user ==3:
    print ("3 is selected")
    match_face()         