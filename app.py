import cv2
import os
from flask import Flask,request,render_template
from datetime import date
from datetime import datetime
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics.pairwise import cosine_similarity,cosine_distances
import pandas as pd
import joblib


#### Defining Flask App
app = Flask(__name__)
#### Saving Date today in 2 different formats
def datetoday():
    return date.today().strftime("%m_%d_%y")
def datetoday2():
    return date.today().strftime("%d-%B-%Y")
#### Initializing VideoCapture object to access WebCam
face_detector = cv2.CascadeClassifier(r'C:\Users\DELL\Desktop\new1\static\haarcascade_frontalface_alt.xml')
cap = cv2.VideoCapture(1)
#### If these directories don't exist, create them
if not os.path.isdir('Attendance'):
    os.makedirs('Attendance')
if not os.path.isdir('static/faces'):
    os.makedirs('static/faces')
if f'Attendance-{datetoday()}.csv' not in os.listdir('Attendance'):
    with open(f'Attendance/Attendance-{datetoday()}.csv','w') as f:
        f.write('Name,Date,Roll,Start Time,End Time')
#### get a number of total registered users
def totalreg():
    return len(os.listdir('static/faces'))
#### extract the face from an image
def extract_faces(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    face_points = face_detector.detectMultiScale(gray, 1.3, 5)
    return face_points
#### Identify face using ML model
def identify_face(facearray): 
    model = joblib.load(r'static\face_recognition_model.pkl')
    return model.predict(facearray)
#### A function which trains the model on all the faces available in faces folder
def train_model():
    faces = []
    labels = []
    userlist = os.listdir('static/faces')
    for user in userlist:
        for imgname in os.listdir(f'static/faces/{user}'):
            img = cv2.imread(f'static/faces/{user}/{imgname}')
            resized_face = cv2.resize(img, (50, 50))
            faces.append(resized_face.ravel())
            labels.append(user)
    faces = np.array(faces)
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(faces,labels)
    joblib.dump(knn,'static/face_recognition_model.pkl')
#### Extract info from today's attendance file in attendance folder
def extract_attendance():
    try:
        df = pd.read_csv(f'Attendance/Attendance-{datetoday()}.csv')
        names = df['Name']
        date=df['Date']
        rolls = df['Roll']
        times = df['Start Time']
        # pw=df['password']
        # dept=df['department']
        etime = df['End Time']
        l = len(df)
        return names,date,rolls,times,etime,l
    except pd.errors.EmptyDataError:
        print("Error: File contains no data or columns.")
        return None, None, None, None, 0


#### Add Attendance of a specific user
def add_attendance(name):
    username = name.split('_')[0]
    userid = name.split('_')[1]
    current_date=date.today()
    current_time = datetime.now().strftime("%H:%M:%S")
    
    try:
        df = pd.read_csv(f'Attendance/Attendance-{datetoday()}.csv')
    except pd.errors.EmptyDataError:
        print("Error: File contains no data or columns.")
        return
    
    if int(userid) not in list(df['Roll']):
        with open(f'Attendance/Attendance-{datetoday()}.csv','a') as f:
            f.write(f'\n{username},{current_date},{userid},{current_time}')
    else:
        idx = df[df['Roll']==int(userid)].index.values[0]
        df.loc[idx, 'End Time'] = current_time
        df.to_csv(f'Attendance/Attendance-{datetoday()}.csv', index=False)



################## ROUTING FUNCTIONS #########################

#### Our main page
#@app.route('/')
#def home():
    #names,date,rolls,times,etime,l = extract_attendance()    
    #return render_template('home.html',names=names,date=date,rolls=rolls,times=times,etime=etime,l=l,totalreg=totalreg(),datetoday2=datetoday2()) 
#### This function will run when we click on Take Attendance Button
@app.route('/start',methods=['GET'])
def start():
    if 'face_recognition_model.pkl' not in os.listdir('static'):
        return render_template('home.html',totalreg=totalreg(),datetoday2=datetoday2(),mess='There is no trained model in the static folder. Please add a new face to continue.') 
    d={}
    #unknown_person={}
    cap = cv2.VideoCapture(1)
    #cap = cv2.VideoCapture(1,cv2.CAP_DSHOW)
    #cap = cv2.VideoCapture(1,cv2.WINDOW_NORMAL)
    #cv2.namedWindow('frame' , cv2.CAP_DSHOW)
    #cv2.namedWindow('frame', cv2.WINDOW_NORMAL)
    
    while True:
        ret,frame = cap.read()
        faces = extract_faces(frame) 
        authorized_users = [r'C:/Users/DELL/Desktop/new1/static/faces']
        #threshold=0.10
        if len(faces) > 0:
            (x, y, w, h) = faces[0]
            center=(int(x+w/2),int(y+h/2))
            radius=int(min(w,h))
            cv2.circle(frame,center,radius, (0, 0, 255), 8)
            face = cv2.resize(frame[y:y+h, x:x+w], (50, 50))
            identified_person = identify_face(face.reshape(1, -1))[0]
            identify_faces=cv2.imread(r'static/faces/' + identified_person + '/4.jpg')
            identify_faces = cv2.resize(identify_faces, (50, 50)).reshape(1, -1)
            face = face.reshape(1, -1)
            similarity = cosine_distances(identify_faces, face)[0][0]
            print(similarity)
            if similarity<0.14:
                if identify_faces in authorized_users:
                    d['Status']='Authorized successfull'
                    add_attendance(identified_person)
                    cv2.putText(frame, f'Authorized: {identified_person}', (400,300), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                else:
                    cv2.putText(frame,
                            f'Unknown person:', (200, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        #else:
               # cv2.putText(frame,
                            #f'Unknown person:{unknown_person}', (200, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
                # if identified_person in authorized_users:
                #     manual_entry = input("Unrecognized person. Would you like to manually enter details for attendance? (y/n): ")
                #     manual_entry.lower() == 'y'
                #     manual_name = input("Enter name: ")
                #     manual_roll = input("Enter roll number: ")
                #     add_attendance(identified_person)

                # for user_folder in authorized_users:
                #     user_path = os.path.join(user_folder, manual_name,manual_roll)
                #     if os.path.exists(user_path):
                #             add_attendance(manual_name)  # Add attendance for the manually entered user
                #             print(f"Attendance added for {manual_name} (Manually entered).")
                #             cv2.putText(frame, f'Authorized (Manual): {manual_name}', (400, 350), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                #             break
                # else:
                #     print("Invalid manual entry. User does not exist in authorized users.")

                #sleep_duration = 10 
        #         #print(f"Unknown person detected. Entering sleep mode for {sleep_duration} seconds.")

        cv2.imshow('Attendance',frame)
        if cv2.waitKey(1)& 0xFF==ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()   
    #return render_template('home.html',names=names,date=date,rolls=rolls,times=times,etime=etime,l=l,totalreg=totalreg(),datetoday2=datetoday2()) 
    print(d)
    return(d)
#### This function will run when we click on Take Attendance Buttons

# #### Our main function which runs the Flask App
if __name__ == '__main__':
     app.run(debug=True)
