from tkinter import *
from PIL import Image, ImageTk
import cv2



class FaceRecogApp:
    def __init__(self, root):
        self.root = root
        self.root.geometry("800x600+200+100")
        self.root.title("Automatic Face Recognition System")
        # Title
        title_lbl = Label(self.root, text="Automatic Face Recognition System", font=("Helvetica", 24, "bold"), bg="darkblue", fg="white")
        title_lbl.place(x=0, y=0, width=800, height=70)
        # Display the Image (UI.png) on the UI for decoration or as a sample
        img_top = Image.open("UI.png")
        img_top = img_top.resize((800, 600), Image.LANCZOS)
        self.photoimg_top = ImageTk.PhotoImage(img_top)
        img_label = Label(self.root, image=self.photoimg_top, bg="lightgray")
        img_label.place(x=10, y=80, width=780, height=510)
        # Button for Face Recognition
        recognize_btn = Button(self.root, text="Recognize", cursor="hand2", command=self.face_recog, font=("Helvetica", 18, "bold"), bg="darkgreen", fg="white")
        recognize_btn.place(x=299, y=520, width=200, height=40)



    def face_recog(self):
        def detect_faces(img, faceCascade):
            gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            # Adjusted scaleFactor and minNeighbors for improved detection
            faces = faceCascade.detectMultiScale(gray_image, scaleFactor=1.05, minNeighbors=8)
            return faces, gray_image
        def recognize_faces(img, gray_image, faces, known_face, known_name):
            for (x, y, w, h) in faces:
                roi_gray = gray_image[y:y + h, x:x + w]
                result = cv2.matchTemplate(roi_gray, known_face, cv2.TM_CCOEFF_NORMED)
                min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)

                if max_val > 0.5:  # Adjusted threshold to be more lenient
                    label = known_name
                    color = (0, 255, 0)  # Green for known face
                else:
                    label = "Unknown"
                    color = (0, 0, 255)  # Red for unknown face
                cv2.rectangle(img, (x, y), (x + w, y + h), color, 3)
                cv2.putText(img, label, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
            return img
        # Load Haar cascade for face detection
        faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        # Load known face and convert it to grayscale
        known_face_img = cv2.imread("Aniket.png")
        gray_known_face = cv2.cvtColor(known_face_img, cv2.COLOR_BGR2GRAY)
        known_faces = faceCascade.detectMultiScale(gray_known_face, scaleFactor=1.1, minNeighbors=5)
        if len(known_faces) == 0:
            print("No face detected in the known face image")
            return
        else:
            (x, y, w, h) = known_faces[0]
            known_face = gray_known_face[y:y + h, x:x + w]
            known_name = "Aniket"
        video_capture = cv2.VideoCapture(0)
        while True:
            ret, img = video_capture.read()
            if not ret:
                break
            # Step 1: Detect faces in the frame
            faces, gray_image = detect_faces(img, faceCascade)
            # Step 2: Recognize the detected faces
            if len(faces) > 0:
                img = recognize_faces(img, gray_image, faces, known_face, known_name)
            else:
                cv2.putText(img, "No Face Detected", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            cv2.imshow("Face Recognition", img)
            if cv2.waitKey(1) == 13:  # Press 'Enter' key to exit
                break
        video_capture.release()
        cv2.destroyAllWindows()



if __name__ == "__main__":
    root = Tk()
    app = FaceRecogApp(root)
    root.mainloop()
