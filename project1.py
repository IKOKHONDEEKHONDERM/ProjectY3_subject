import pygame
import cv2 
import mediapipe as mp
import numpy as np
import sys
import threading

# กำหนดค่าพื้นฐาน Mediapipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# เริ่มต้น Pygame
pygame.init()
pygame.display.set_caption("Body Detection GUI")

# กำหนดค่าหน้าจอ
screen_width, screen_height = 640, 480
screen = pygame.display.set_mode((screen_width, screen_height))

# เริ่มต้นกล้อง OpenCV
cap = cv2.VideoCapture(0)

# กำหนดสี
white = (255, 255, 255)
blue = (0, 102, 204)
black = (0, 0, 0)

# ฟังก์ชันวาดปุ่ม
def draw_button(screen, text, x, y, width, height, color, text_color):
    font = pygame.font.Font(None, 36)
    button_rect = pygame.Rect(x, y, width, height)
    pygame.draw.rect(screen, color, button_rect)
    text_surface = font.render(text, True, text_color)
    screen.blit(text_surface, (x + (width - text_surface.get_width()) // 2, 
                               y + (height - text_surface.get_height()) // 2))
    return button_rect

# ฟังก์ชันตรวจจับการยกแขนขึ้น
def is_raising_hand(landmarks):
    if landmarks and landmarks.landmark:
        right_shoulder = landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER]
        right_wrist = landmarks.landmark[mp_pose.PoseLandmark.RIGHT_WRIST]
        return right_wrist.y < right_shoulder.y
    return False

def is_touching_shoulders(landmarks):
    if landmarks and landmarks.landmark:
        right_wrist = landmarks.landmark[mp_pose.PoseLandmark.RIGHT_WRIST]
        left_wrist = landmarks.landmark[mp_pose.PoseLandmark.LEFT_WRIST]
        right_shoulder = landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER]
        left_shoulder = landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER]

        # ตรวจสอบว่าข้อมือทั้งสองข้างอยู่ใกล้ไหล่ด้านเดียวกัน
        return (
            abs(right_wrist.x - right_shoulder.x) < 0.1 and
            abs(right_wrist.y - right_shoulder.y) < 0.1 and
            abs(left_wrist.x - left_shoulder.x) < 0.1 and
            abs(left_wrist.y - left_shoulder.y) < 0.1
        )
    return False

def is_raising_left_hand(landmarks):
    if landmarks and landmarks.landmark:
        left_shoulder = landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER]
        left_wrist = landmarks.landmark[mp_pose.PoseLandmark.LEFT_WRIST]
        # เช็คว่ามือซ้ายอยู่สูงกว่าไหล่ซ้าย
        return left_wrist.y < left_shoulder.y
    return False

def is_crossing_arms(landmarks):
    if landmarks and landmarks.landmark:
        right_wrist = landmarks.landmark[mp_pose.PoseLandmark.RIGHT_WRIST]
        left_wrist = landmarks.landmark[mp_pose.PoseLandmark.LEFT_WRIST]
        right_shoulder = landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER]
        left_shoulder = landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER]

        # ตรวจสอบว่าข้อมือซ้ายและขวาอยู่ใกล้กันบริเวณกลางลำตัว
        return (
            abs(right_wrist.x - left_wrist.x) < 0.1 and  # ระยะห่างแกน X น้อย
            right_wrist.y > left_shoulder.y and          # ข้อมือขวาอยู่ใต้ไหล่ซ้าย
            left_wrist.y > right_shoulder.y             # ข้อมือซ้ายอยู่ใต้ไหล่ขวา
        )
    return False

def is_hands_on_hips_and_legs_apart(landmarks):
    if landmarks and landmarks.landmark:
        # จุดที่ต้องการ
        right_wrist = landmarks.landmark[mp_pose.PoseLandmark.RIGHT_WRIST]
        left_wrist = landmarks.landmark[mp_pose.PoseLandmark.LEFT_WRIST]
        right_hip = landmarks.landmark[mp_pose.PoseLandmark.RIGHT_HIP]
        left_hip = landmarks.landmark[mp_pose.PoseLandmark.LEFT_HIP]
        right_knee = landmarks.landmark[mp_pose.PoseLandmark.RIGHT_KNEE]
        left_knee = landmarks.landmark[mp_pose.PoseLandmark.LEFT_KNEE]
        right_shoulder = landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER]
        left_shoulder = landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER]

        # เงื่อนไขการเท้าเอว
        hands_on_hips = (
            abs(right_wrist.x - right_hip.x) < 0.1 and
            abs(left_wrist.x - left_hip.x) < 0.1 and
            right_wrist.y > right_shoulder.y and
            left_wrist.y > left_shoulder.y
        )

        # เงื่อนไขการกางขา
        legs_apart = abs(right_knee.x - left_knee.x) > 0.4  # ระยะห่างเข่าขึ้นอยู่กับภาพจริง

        return hands_on_hips and legs_apart
    return False

# ฟังก์ชันเล่นเสียงค้างไว้
def play_beep():
    while hand_raised:
        pygame.mixer.init()
        pygame.mixer.music.load("D:\WORK\Year-3-2\design\do.wav")  # ใส่ path ไฟล์เสียงของคุณ
        pygame.mixer.music.play()
        pygame.time.wait(1000)

def play_r():
    while hand_raised:
        pygame.mixer.init()
        pygame.mixer.music.load("D:\WORK\Year-3-2\design\RE.wav")  # ใส่ path ไฟล์เสียงของคุณ
        pygame.mixer.music.play()
        pygame.time.wait(1000)

def play_MI():
    while arms_crossed:
        pygame.mixer.init()
        pygame.mixer.music.load("D:\WORK\Year-3-2\design\MI.wav")  # ใส่ path ไฟล์เสียงของคุณ
        pygame.mixer.music.play()
        pygame.time.wait(1000) 

def play_sound_hands_hips_legs_apart():
    while hands_hips_legs_apart:
        pygame.mixer.init()
        pygame.mixer.music.load("D:\WORK\Year-3-2\design\FA.wav")  # ใส่ path ไฟล์เสียง FA ของคุณ
        pygame.mixer.music.play()
        pygame.time.wait(1000)  # รอ 1 วินาที


# สถานะการทำงานของกล้อง
camera_active = False
hand_raised = False
jumping_jack_detected = False

running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:  # ปิดโปรแกรม
            running = False
        elif event.type == pygame.MOUSEBUTTONDOWN:  # ตรวจจับการคลิกเมาส์
            if button_rect.collidepoint(event.pos):
                camera_active = not camera_active

    # ล้างหน้าจอ
    screen.fill(white)

    # วาดปุ่มเปิด/ปิดกล้อง
    button_color = blue if not camera_active else (200, 0, 0)
    button_text = "Start Camera" if not camera_active else "Stop Camera"
    button_rect = draw_button(screen, button_text, 220, 400, 200, 50, button_color, white)

    # แสดงผลจากกล้องในหน้าต่าง Pygame
    if camera_active:
        ret, frame = cap.read()
        if ret:
            # ประมวลผลภาพด้วย Mediapipe
            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(image_rgb)
            if results.pose_landmarks:
                mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

                # ตรวจจับการยกแขน
                if is_raising_hand(results.pose_landmarks):
                    if not hand_raised:
                        hand_raised = True
                        threading.Thread(target=play_beep, daemon=True).start()
                else:
                    hand_raised = False
                
                arms_crossed = False

# ในลูป while หลัก
                if is_crossing_arms(results.pose_landmarks):
                    if not arms_crossed:
                        arms_crossed = True
                        print("Cross Arms detected!")
                        threading.Thread(target=play_MI, daemon=True).start()
                else:
                    arms_crossed = False

                hands_hips_legs_apart = False

# ในลูป while หลัก
                if is_hands_on_hips_and_legs_apart(results.pose_landmarks):
                    if not hands_hips_legs_apart:
                        hands_hips_legs_apart = True
                        print("Hands on hips and legs apart detected!")
                        threading.Thread(target=play_sound_hands_hips_legs_apart, daemon=True).start()
                else:
                    hands_hips_legs_apart = False


                if is_touching_shoulders(results.pose_landmarks):
                    if not shoulders_touched:
                        shoulders_touched = True
                        print("Shoulders touched!")
                        threading.Thread(target=play_r, daemon=True).start()
                else:
                    shoulders_touched = False

            # แปลงภาพ BGR -> RGB -> Surface (สำหรับแสดงใน Pygame)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = np.rot90(frame)  # หมุนภาพ 90 องศา
            frame_surface = pygame.surfarray.make_surface(frame)
            screen.blit(frame_surface, (0, 0))  # แสดงผลภาพบนหน้าจอ Pygame

    pygame.display.flip()

# ปิดทรัพยากร
cap.release()
cv2.destroyAllWindows()
pygame.quit()
sys.exit()