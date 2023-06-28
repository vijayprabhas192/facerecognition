import cv2
from flask import Flask,request ,render_template,jsonify
import os
from datetime import date
from datetime import datetime
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics.pairwise import cosine_similarity,cosine_distances
import pandas as pd
import joblib
import numpy as np

#### Defining Flask App
app = Flask(__name__)

#### Saving Date today in 2 different formats
def datetoday():
    return date.today().strftime("%m_%d_%y")
def datetoday2():
    return date.today().strftime("%d-%B-%Y")

#### Initializing VideoCapture object to access WebCam
face_detector = cv2.CascadeClassifier(r'static\haarcascade_frontalface_alt.xml')
#cap = cv2.VideoCapture(0)

#### If these directories don't exist, create them
if not os.path.isdir('Attendance'):
    
    os.makedirs('Attendance')
if not os.path.isdir('static/faces'):
    os.makedirs('static/faces')
if f'Attendance-{datetoday()}.csv' not in os.listdir('Attendance'):
    with open(f'Attendance/Attendance-{datetoday()}.csv','w') as f:
        f.write('Name,Date,Roll,Start Time,End Time')
if not os.path.isdir('visitor_Attendance'):    
    os.makedirs('visitor_Attendance')
if not os.path.isdir('static/visitor_faces'):
    os.makedirs('static/visitor_faces')
if f'visitor_Attendance-{datetoday()}.csv' not in os.listdir('Attendance'):
    with open(f'visitor_Attendance/Attendance-{datetoday()}.csv','w') as f:
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

#### Extract info from today's attendance file in attendance folder
def extract_attendance():
    try:
        df = pd.read_csv(f'Attendance/Attendance-{datetoday()}.csv')
        names = df['Name']
        date=df['Date']
        rolls = df['Roll']
        times = df['Start Time']
        etime = df['End Time']
        l = len(df)
        return names,date,rolls,times,etime,l
    except pd.errors.EmptyDataError:
        print("Error: File contains no data or columns.")
        return None, None, None, None, 0

def extract_attendance_visitor():
    try:
        df = pd.read_csv(f'Visitor_Attendance/Attendance-{datetoday()}.csv')
        print(df)
        visitor_names = df['visitor_name']
        date=df['Date']
        rolls = df['mobile_number']
        visitorpurpose=df['visitor_purpose']
        times = df['Entry Time']
        etime = df['Exit Time']
        l = len(df)
        return visitor_names,date,rolls,visitorpurpose,times,etime,l
    except pd.errors.EmptyDataError:
        print("Error: File contains no data or columns.")
        return None,None,None,None,None,None,0
#### A function which trains the model on all the faces available in faces folder
def train_model():
    faces = []
    labels = []
    userlist = os.listdir('static/faces')
    for user in userlist:
        for imgname in os.listdir(f'static/faces/{user}'):
            img = cv2.imread(f'static/faces/{user}/{imgname}')
            resized_face = cv2.resize(img, (50,50))
            faces.append(resized_face.ravel())
            labels.append(user)
    faces = np.array(faces)
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(faces,labels)
    joblib.dump(knn,r"static\face_recognition_model.pkl")

#### Our main page
def train_model():
    faces = []
    labels = []
    userlist = os.listdir('static/visitor_faces')
    for user in userlist:
        for imgname in os.listdir(f'static/visitor_faces/{user}'):
            img = cv2.imread(f'static/visitor_faces/{user}/{imgname}')
            resized_face = cv2.resize(img, (50,50))
            faces.append(resized_face.ravel())
            labels.append(user)
    faces = np.array(faces)
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(faces,labels)
    joblib.dump(knn,r"static\face_recognition_model.pkl")

#### This function will run when we add a new user
userimagedir=[]
import random
@app.route('/add',methods=['GET','POST'])
def add():
    f=open(f'Attendance/Attendance-{datetoday()}.csv','a')
    f.close()
    userimagefolder='static/faces/'+str(random.randint(0,999))
    userimagedir.append(userimagefolder)
    if not os.path.isdir(userimagefolder):
        os.makedirs(userimagefolder)
    cap = cv2.VideoCapture(1)
    i,j = 0,0
    while 1:
        _,frame = cap.read()
        faces = extract_faces(frame)
        for (x,y,w,h) in faces:
            center=(int(x+w/2),int(y+h/2))
            radius=int(min(w,h)/2)
            cv2.circle(frame,center,radius, (0, 0, 255), 6)
            cv2.putText(frame,f'Images Captured: {i}/50',(30,30),cv2.FONT_HERSHEY_SIMPLEX,1,(0, 255, 0),2,cv2.LINE_AA)
            if j%10==0:
                name = str(i)+'.jpg'
                cv2.imwrite(userimagefolder+'/'+name,frame[y:y+h,x:x+w])
                i+=1
            j+=1         
        if j==500:
            break
        cv2.imshow('Adding new User',frame)
        if cv2.waitKey(1)==27:
            break
    cap.release()
    cv2.destroyAllWindows()
    #add csv file
    names,date,rolls,times,etime,l = extract_attendance()
    return {"sucess":True}

@app.route('/add_data',methods=['GET','POST'])
def add_data():
    
    newusername = str(request.args['username'])
    newuserid = str(request.args['userid'])
    
    from datetime import date
    current_date=date.today()
    current_time = datetime.now().strftime("%H:%M:%S")
    new_name = newusername+'_'+str(newuserid)
    print("dir  :",userimagedir,"       :","/".join(userimagedir[-1].split('/')[:-1])+'/'+ new_name)
    
    os.rename(userimagedir[-1],"/".join(userimagedir[-1].split('/')[:-1])+'/'+ new_name)
    
    if (newusername,newuserid) not in list(['Name','Roll']):
        with open(f'Attendance/Attendance-{datetoday()}.csv','a') as f:
            #f.write(f'\n{newusername},{newuserid}')
            f.write(f'\n{newusername},{current_date},{newuserid},{current_time}')
    print('Training Model')
    train_model()
    #newusername,newuserid=add_attendance(name) 
    names,date,rolls,times,etime,l = extract_attendance()
    #return render_template('home.html',names=names,date=date,rolls=rolls,times=times,etime=etime,l=l,totalreg=totalreg(),datetoday2=datetoday2())
    d={}
    d['outout']= 'Form Submission is Success. '
    return d
    # response = {'success' : True}
    # return jsonify(response)
####################################################################
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
######################################################
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
            if similarity>0.14:
                #if identify_faces in authorized_users:
                    d['Status']='Authorized successfull'
                    add_attendance(identified_person)
                    cv2.putText(frame, f'Authorized: {identified_person}', (400,300), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            else:
                    cv2.putText(frame,
                            f'Unknown person:', (200, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        cv2.imshow('Attendance',frame)
        if cv2.waitKey(1)& 0xFF==ord('q'):
                break
    cap.release()
    cv2.destroyAllWindows()   
    #return render_template('home.html',names=names,date=date,rolls=rolls,times=times,etime=etime,l=l,totalreg=totalreg(),datetoday2=datetoday2()) 
    print(d)
    return(d)
import random
userimagedir=[]
@app.route('/add_visitor',methods=['GET','POST'])
def add_visitor():
    f=open(f'Visitor_Attendance/Attendance-{datetoday()}.csv','a')
    f.close()
    userimagefolder='static/visitor_faces/'+str(random.randint(0,999))
   
    userimagedir.append(userimagefolder)
    if not os.path.isdir(userimagefolder):
        os.makedirs(userimagefolder)
    cap = cv2.VideoCapture(0)
    i,j = 0,0
    while 1:
        _,frame = cap.read()
        faces = extract_faces(frame)
        for (x,y,w,h) in faces:
            center=(int(x+w/2),int(y+h/2))
            radius=int(min(w,h)/2)
            cv2.circle(frame,center,radius, (0, 0, 255), 6)
            cv2.putText(frame,f'Images Captured: {i}/5',(30,30),cv2.FONT_HERSHEY_SIMPLEX,1,(255, 0, 20),2,cv2.LINE_AA)
            if j%10==0:
                name = str(i)+'.jpg'
                cv2.imwrite(userimagefolder+'/'+name,frame[y:y+h,x:x+w])
                i+=1
            j+=1         
        if j==50:
            break
        cv2.imshow('Adding new User',frame)
        if cv2.waitKey(1)==27:
            break
    cap.release()
    cv2.destroyAllWindows()
    #add csv file
    names,date,rolls,times,visitorpupose,etime,l = extract_attendance()
    return {"sucess":True}

@app.route('/add_data_visitor',methods=['GET','POST'])
def add_data_visitor():
    
    newusername = str(request.args['username'])
    phone_number = str(request.args['mobile_number'])
    visitor = str(request.args['visitor_purpose'])
    from datetime import date
    current_date=date.today()
    current_time = datetime.now().strftime("%H:%M:%S")
    new_name = newusername+''+str(phone_number)+''+str(visitor)
    print("dir  :",userimagedir,"       :","/".join(userimagedir[-1].split('/')[:-1])+'/'+ new_name)
    
    os.rename(userimagedir[-1],"/".join(userimagedir[-1].split('/')[:-1])+'/'+ new_name)
    
    if (newusername,phone_number) not in list(['Name','Mobilenumber']):
        with open(f'VisitorAttendance/Attendance-{datetoday()}.csv','a') as f:
            #f.write(f'\n{newusername},{newuserid}')
            f.write(f'\n{newusername},{current_date},{phone_number},{current_time},{visitor}')
    print('Training Model')
    train_model()
    #newusername,newuserid=add_attendance(name) 
    names,date,rolls,times,visitorpupose,etime,l = extract_attendance()
    #return render_template('home.html',names=names,date=date,rolls=rolls,times=times,etime=etime,l=l,totalreg=totalreg(),datetoday2=datetoday2())
    d={}
    d['output']= 'Form Submission is Success. '
    return d

#### Our main function which runs the Flask App
if __name__ == '__main__':
    app.run(debug=True)


