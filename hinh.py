import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog, ttk, simpledialog, messagebox
from PIL import Image, ImageTk
import face_recognition
import os
import pickle

# Thư mục lưu trữ dữ liệu khuôn mặt đã biết
KNOWN_FACES_DIR = "known_faces"
ENCODINGS_FILE = "known_encodings.pkl"

# Tạo thư mục nếu chưa tồn tại
if not os.path.exists(KNOWN_FACES_DIR):
    os.makedirs(KNOWN_FACES_DIR)

# Tải các mã hóa khuôn mặt đã biết
known_face_encodings = []
known_face_names = []

if os.path.exists(ENCODINGS_FILE):
    with open(ENCODINGS_FILE, 'rb') as f:
        data = pickle.load(f)
        known_face_encodings = data['encodings']
        known_face_names = data['names']

root = tk.Tk()
root.title("YOLOv3 Object Detection with Face Recognition")
root.geometry("300x400")

net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

# Load class names from coco.names
classes = []
with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]

selected_class = tk.StringVar()  # Variable to store selected class

def choose_video():
    video_path = filedialog.askopenfilename(filetypes=[("Video files", "*.mp4;*.avi;*.mov")])
    if video_path:
        detect_objects_in_video(video_path)

def detect_objects_in_video(video_path):
    cap = cv2.VideoCapture(video_path)
    detect_objects(cap)

def detect_objects_in_camera():
    cap = cv2.VideoCapture(0)  # Open the webcam (0 is usually the default camera)
    detect_objects(cap)

def detect_objects(cap):
    global known_face_encodings, known_face_names  # To access and modify global variables
    target_class = selected_class.get()  # Get the selected target class from dropdown

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        height, width, channels = frame.shape
        blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
        net.setInput(blob)
        outs = net.forward(output_layers)

        class_ids = []
        confidences = []
        boxes = []

        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > 0.5:
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)

                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)

                    if classes[class_id] == target_class or target_class == "All":
                        boxes.append([x, y, w, h])
                        confidences.append(float(confidence))
                        class_ids.append(class_id)

        indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
        font = cv2.FONT_HERSHEY_PLAIN

        for i in range(len(boxes)):
            if i in indexes:
                x, y, w, h = boxes[i]
                label = str(classes[class_ids[i]])

                # Nếu đối tượng là "person", tiến hành nhận diện khuôn mặt
                if label == "person":
                    # Trích xuất vùng chứa khuôn mặt
                    person_roi = frame[y:y+h, x:x+w]
                    rgb_person = cv2.cvtColor(person_roi, cv2.COLOR_BGR2RGB)

                    # Sử dụng face_recognition để tìm vị trí khuôn mặt
                    face_locations = face_recognition.face_locations(rgb_person)
                    face_encodings = face_recognition.face_encodings(rgb_person, face_locations)

                    for face_encoding, face_location in zip(face_encodings, face_locations):
                        # So sánh với các khuôn mặt đã biết
                        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
                        name = "Person"

                        face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
                        best_match_index = np.argmin(face_distances) if len(face_distances) > 0 else -1

                        if best_match_index >= 0 and matches[best_match_index]:
                            name = known_face_names[best_match_index]

                        # Vẽ khung và tên
                        top, right, bottom, left = face_location
                        # Điều chỉnh tọa độ khuôn mặt theo tọa độ của bounding box
                        top += y
                        right += x
                        bottom += y
                        left += x

                        cv2.rectangle(frame, (left, top), (right, bottom), (255, 0, 0), 2)
                        cv2.putText(frame, name, (left, top - 10), font, 1, (255, 0, 0), 2)
                else:
                    color = (0, 255, 0)
                    cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                    cv2.putText(frame, label, (x, y - 10), font, 2, color, 2)

        cv2.imshow("YOLOv3 Object Detection", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

def save_face():
    # Sử dụng camera để chụp khuôn mặt và lưu trữ
    cap = cv2.VideoCapture(0)
    messagebox.showinfo("Hướng dẫn", "Nhấn 's' để lưu khuôn mặt hoặc 'q' để hủy.")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        face_locations = face_recognition.face_locations(rgb_frame)
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

        for (top, right, bottom, left) in face_locations:
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)

        cv2.imshow("Save Face - Press 's' to save, 'q' to quit", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('s'):
            if len(face_encodings) == 0:
                messagebox.showwarning("Cảnh báo", "Không tìm thấy khuôn mặt. Vui lòng thử lại.")
                continue
            # Giả sử chỉ có một khuôn mặt trong khung hình
            face_encoding = face_encodings[0]
            name = simpledialog.askstring("Nhập Tên", "Nhập tên của người này:")

            if name:
                known_face_encodings.append(face_encoding)
                known_face_names.append(name)

                # Lưu dữ liệu vào tệp
                with open(ENCODINGS_FILE, 'wb') as f:
                    data = {'encodings': known_face_encodings, 'names': known_face_names}
                    pickle.dump(data, f)

                # Lưu hình ảnh khuôn mặt
                face_image = rgb_frame[top:bottom, left:right]
                pil_image = Image.fromarray(face_image)
                pil_image = pil_image.resize((128, 128))  # Resize để tiết kiệm không gian
                face_filename = os.path.join(KNOWN_FACES_DIR, f"{name}.jpg")
                pil_image.save(face_filename)

                messagebox.showinfo("Thành công", f"Đã lưu khuôn mặt của {name}.")
            else:
                messagebox.showwarning("Cảnh báo", "Tên không được để trống.")

        elif key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# Tạo dropdown menu để chọn đối tượng
label_select = tk.Label(root, text="Chọn đối tượng:")
label_select.pack(pady=10)

# Thêm tùy chọn "All" để nhận diện tất cả các đối tượng
class_dropdown = ttk.Combobox(root, textvariable=selected_class, values=["All"] + classes, state="readonly", height=15)
class_dropdown.pack(pady=10)
class_dropdown.set("All")  # Đặt tùy chọn mặc định là "All"

# Tạo nút để chọn video và bắt đầu nhận diện
btn_select_video = tk.Button(root, text="Chọn Video", command=choose_video, height=2, width=20)
btn_select_video.pack(pady=10)

# Tạo nút để sử dụng webcam cho nhận diện
btn_camera = tk.Button(root, text="Dùng Camera", command=detect_objects_in_camera, height=2, width=20)
btn_camera.pack(pady=10)

# Tạo nút để lưu khuôn mặt
btn_save_face = tk.Button(root, text="Lưu Khuôn Mặt", command=save_face, height=2, width=20, bg="yellow")
btn_save_face.pack(pady=10)

root.mainloop()