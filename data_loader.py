import cv2
import numpy as np

def load_data(load_train=True, load_test=True):
    x_train = []
    y_train = []
    x_test = []
    y_test = []

    dictionary = {"A":1,"B":2,"C":3,"D":4,"E":5,"F":6,"G":7,"H":8,"I":9,"J":10,"K":11,"L":12,"M":13,"N":14,"O":15,"P":16,
    "Q":17,"R":18,"S":19,"T":20,"U":21,"V":22,"W":23,"X":24,"Y":25,"Z":26,
    "del":27,"space":28,"nothing":29}

    values = list(dictionary.values())
    keys = list(dictionary.keys())

    if load_train:

        string = "asl-alphabet/asl_alphabet_train/"
        folder = ""
        image = ""
        ext = ".jpg"

        print("Loading Training Data..............")

        for folders in keys:
            for images in range(1,3001):
                load = string+folders+"/"+folders+str(images)+ext
                img = cv2.imread(load)
                if str(type(img)) != "<class 'NoneType'>":
                    img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
                    img = cv2.resize(img, (50,50))
                    x_train.append(img)
                    y_train.append(dictionary[folders])

        x_train = np.array(x_train)
        y_train = np.array(y_train)
        print("Data Loaded................")

    if load_test:
        string = "asl-alphabet/asl_alphabet_test/"
        image = ""
        ext = ".jpg"

        print("Loading Test Data.........")

        for images in keys:
            load = string+images+"_test"+ext
            img = cv2.imread(load)
            if str(type(img)) != "<class 'NoneType'>":
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                img = cv2.resize(img, (50,50))
                x_test.append(img)
                y_test.append(dictionary[images])

        x_test = np.array(x_test)
        y_test = np.array(y_test)

        print("Data Loaded.............")

        return (x_train,y_train,x_test,y_test)

    return (x_train, y_train)
