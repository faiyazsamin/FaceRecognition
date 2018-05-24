import cv2
import numpy as np
import os

subjects = ["","Mama","Samin","Delwar"]


def detect_faces(colored_img, scaleFactor=1.06):

    img_copy = colored_img.copy()
    gray = cv2.cvtColor(img_copy, cv2.COLOR_BGR2GRAY)
    f_cascade = cv2.CascadeClassifier('data/lbpcascade_frontalface.xml')
    faces = f_cascade.detectMultiScale(gray, scaleFactor=scaleFactor, minNeighbors=5);
    if len(faces) == 0:
        return None, None
    (x, y, w, h) = faces[0]
    return gray[y:y+w, x:x+h], faces[0]


def prepare_training_data(data_folder_path):

    dirs = os.listdir(data_folder_path)
    faces = []
    labels = []

    for dir_name in dirs:

        if not dir_name.startswith("s"):
            continue

        label = int(dir_name.replace("s", ""))
        subject_dir_path = data_folder_path + "/" + dir_name
        subject_images_names = os.listdir(subject_dir_path)

        for image_name in subject_images_names:
            if image_name.startswith("."):
                continue

            image_path = subject_dir_path + "/" + image_name
            image = cv2.imread(image_path)
            cv2.imshow("Training on image...", cv2.resize(image, (400, 500)))
            cv2.waitKey(10)

            face, rect = detect_faces(image)
            if face is not None:
                faces.append(face)
                labels.append(label)

    cv2.destroyAllWindows()
    cv2.waitKey(1)
    cv2.destroyAllWindows()
    print("Total faces: ", len(faces))
    print("Total labels: ", len(labels))

    return faces, labels


def trainData(trainingDataPath, output_path):
    face_recognizer = cv2.face.LBPHFaceRecognizer_create()
    faces, labels = prepare_training_data(trainingDataPath)

    face_recognizer.train(faces, np.array(labels))
    face_recognizer.write(output_path)


def loadTrainedData(path):
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read(path)
    return recognizer


def predictStaticImage(test_img,trainer_file):
    img = test_img.copy()
    face, rect = detect_faces(img)
    lt = loadTrainedData(trainer_file)
    label, confidence = lt.predict(face)
    label_text = subjects[label]
    (x, y, w, h) = rect
    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
    cv2.putText(img, label_text, (rect[0], rect[1] - 5), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 255, 0), 2)
    print("Confidence =",confidence)
    return img

def showImage(image):
    cv2.imshow('Frame', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def camToFile(framesToCapture,output_dir):
    cam = cv2.VideoCapture(1)
    detector = cv2.CascadeClassifier('data/haarcascade_frontalface_alt.xml')
    sampleNum = 0

    while True:
        ret, img = cam.read()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        face = detector.detectMultiScale(gray, 1.5, 5)
        for (x, y, w, h) in face:
            cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)

            sampleNum = sampleNum + 1
            if sampleNum%(100/framesToCapture) == 0:
                print("Frames Captured:", int(sampleNum/(100/framesToCapture)))
                cv2.imwrite(output_dir+"/"+ str(int(sampleNum/(100/framesToCapture))) + ".jpg", gray[y:y + h, x:x + w])

        cv2.imshow('frame', img)
        if cv2.waitKey(100) & 0xFF == ord('q'):
            break
        elif sampleNum >= 100:
            break


def detectFace(trainer_file):
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read(trainer_file)
    faceCascade = cv2.CascadeClassifier("data/haarcascade_frontalface_alt.xml")

    cam = cv2.VideoCapture(1)
    font = cv2.FONT_HERSHEY_DUPLEX
    while True:
        ret, im = cam.read()
        gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        faces = faceCascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5, minSize=(100, 100),
                                             flags=cv2.CASCADE_SCALE_IMAGE)
        for (x, y, w, h) in faces:
            nbr_predicted, conf = recognizer.predict(gray[y:y + h, x:x + w])
            cv2.rectangle(im, (x - 50, y - 50), (x + w + 50, y + h + 50), (0, 225, 0), 2)
            nbr_predicted = subjects[nbr_predicted]
            cv2.putText(im, str(nbr_predicted), (x + 30, y + h + 30), font, 1, (0, 0, 225))  # Draw the text
        cv2.imshow('FaceDetector', im)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cam.release()
    cv2.destroyAllWindows()


trainData('training-data','test.yml')
detectFace('test.yml')
#showImage(predictStaticImage(cv2.imread("test-data/4.jpg"),'test3.yml'))
#camToFile(20,'training-data/s7')