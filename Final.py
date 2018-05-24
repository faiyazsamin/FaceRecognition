import cv2
import numpy as np
import os

subjects = ["", "Samin", "Ramiz Raja", "Sam","Samiha","Ammu"]


def detect_faces(colored_img, scaleFactor=1.06):
    img_copy = colored_img.copy()

    # convert the test image to gray image as opencv face detector expects gray images
    gray = cv2.cvtColor(img_copy, cv2.COLOR_BGR2GRAY)

    f_cascade = cv2.CascadeClassifier('data/lbpcascade_frontalface.xml')

    # let's detect multiscale (some images may be closer to camera than others) images
    faces = f_cascade.detectMultiScale(gray, scaleFactor=scaleFactor, minNeighbors=5);

    if (len(faces) == 0):
        return None, None

    # go over list of faces and draw them as rectangles on original colored img
    #for (x, y, w, h) in faces:
    #    cv2.rectangle(img_copy, (x, y), (x + w, y + h), (0, 255, 0), 2)

    (x, y, w, h) = faces[0]

    return gray[y:y+w, x:x+h], faces[0]

def prepare_training_data(data_folder_path):
    # ------STEP-1--------
    # get the directories (one directory for each subject) in data folder
    dirs = os.listdir(data_folder_path)

    # list to hold all subject faces
    faces = []
    # list to hold labels for all subjects
    labels = []

    # let's go through each directory and read images within it
    for dir_name in dirs:

        # our subject directories start with letter 's' so
        # ignore any non-relevant directories if any
        if not dir_name.startswith("s"):
            continue;

        # ------STEP-2--------
        # extract label number of subject from dir_name
        # format of dir name = slabel
        # , so removing letter 's' from dir_name will give us label
        label = int(dir_name.replace("s", ""))

        # build path of directory containin images for current subject subject
        # sample subject_dir_path = "training-data/s1"
        subject_dir_path = data_folder_path + "/" + dir_name

        # get the images names that are inside the given subject directory
        subject_images_names = os.listdir(subject_dir_path)

        # ------STEP-3--------
        # go through each image name, read image,
        # detect face and add face to list of faces
        for image_name in subject_images_names:

            # ignore system files like .DS_Store
            if image_name.startswith("."):
                continue;

            # build image path
            # sample image path = training-data/s1/1.pgm
            image_path = subject_dir_path + "/" + image_name

            # read image
            image = cv2.imread(image_path)

            # display an image window to show the image
            cv2.imshow("Training on image...", cv2.resize(image, (400, 500)))
            cv2.waitKey(10)

            # detect face
            face, rect = detect_faces(image)

            # ------STEP-4--------
            # for the purpose of this tutorial
            # we will ignore faces that are not detected
            if face is not None:
                # add face to list of faces
                faces.append(face)
                # add label for this face
                labels.append(label)

    cv2.destroyAllWindows()
    cv2.waitKey(1)
    cv2.destroyAllWindows()
    print("Total faces: ", len(faces))
    print("Total labels: ", len(labels))

    return faces, labels


#print total faces and labels


def trainData(trainingDataPath, output_path):
    face_recognizer = cv2.face.LBPHFaceRecognizer_create()
    faces, labels = prepare_training_data(trainingDataPath)
    # train our face recognizer of our training faces
    face_recognizer.train(faces, np.array(labels))
    face_recognizer.write(output_path)
    # faces, lables = prepare_training_data('training-data')
    # trainData(faces,lables,'test3.yml')



def loadTrainedData(path):
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read(path)
    return recognizer


def predictStaticImage(test_img,trainer_file):
    # make a copy of the image as we don't want to chang original image
    img = test_img.copy()
    # detect face from the image
    face, rect = detect_faces(img)

    # predict the image using our face recognizer
    lt = loadTrainedData(trainer_file)
    label, confidence = lt.predict(face)
    # get name of respective label returned by face recognizer
    label_text = subjects[label]

    # draw a rectangle around face detected
    #draw_rectangle(img, rect)
    (x, y, w, h) = rect
    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
    # draw name of predicted person
    #draw_text(img, label_text, rect[0], rect[1] - 5)
    cv2.putText(img, label_text, (rect[0], rect[1] - 5), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 255, 0), 2)
    print("Confidence =",confidence)
    return img

def showImage(image):
    cv2.imshow('Frame', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

#perform a prediction


def camToFile(output_dir):
    cam = cv2.VideoCapture(0)
    #detector = cv2.CascadeClassifier('data/lbpcascade_frontalface.xml')
    detector =  cv2.CascadeClassifier('data/haarcascade_frontalface_alt.xml')
    sampleNum = 0

    while (True):
        ret, img = cam.read()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = detector.detectMultiScale(gray, 1.5, 5)
        for (x, y, w, h) in faces:
            cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)

            # incrementing sample number
            sampleNum = sampleNum + 1
            print("Frames Captured: ", sampleNum)
            # saving the captured face in the dataset folder
            cv2.imwrite(output_dir+"/"+ str(sampleNum) + ".jpg", gray[y:y + h, x:x + w])

        cv2.imshow('frame', img)
        if cv2.waitKey(100) & 0xFF == ord('q'):
            break
        elif sampleNum >= 20:
            break


def detectFace(trainer_file):
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read(trainer_file)
    #cascadePath = "data/lbpcascade_frontalface.xml"
    cascadePath = "data/haarcascade_frontalface_alt.xml"
    faceCascade = cv2.CascadeClassifier(cascadePath);
    path = 'dataSet'

    cam = cv2.VideoCapture(0)
    font = cv2.FONT_HERSHEY_DUPLEX  # Creates a font
    while True:
        ret, im = cam.read()
        gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        faces = faceCascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5, minSize=(100, 100),
                                             flags=cv2.CASCADE_SCALE_IMAGE)
        for (x, y, w, h) in faces:
            nbr_predicted, conf = recognizer.predict(gray[y:y + h, x:x + w])
            cv2.rectangle(im, (x - 50, y - 50), (x + w + 50, y + h + 50), (0, 225, 0), 2)
            nbr_predicted = subjects[nbr_predicted]
            # +"--"+str(conf)
            cv2.putText(im, str(nbr_predicted), (x, y + h), font, 1, (0, 0, 225))  # Draw the text
        cv2.imshow('im', im)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cam.release()
    cv2.destroyAllWindows()


#faces, lables = prepare_training_data('training-data')
trainData('training-data','test3.yml')
#detectFace('test3.yml')
#showImage(predictStaticImage(cv2.imread("test-data/3.jpg"),'test2.yml'))

#camToFile('training-data/s5')