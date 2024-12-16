import pygame
import cv2
import mediapipe as mp
import numpy as np
import sys
import threading
import winsound

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
    # เช็คว่า landmark ของไหล่และมือขวาถูกตรวจพบ
    if landmarks and landmarks.landmark:
        right_shoulder = landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER]
        right_wrist = landmarks.landmark[mp_pose.PoseLandmark.RIGHT_WRIST]
        # เช็คว่ามืออยู่สูงกว่าไหล่
        return right_wrist.y < right_shoulder.y
    return False

# ฟังก์ชันเล่นเสียงค้างไว้
def play_beep():
    while hand_raised:
        winsound.Beep(261,500)  # เล่นโน้ตตัว "โด" (261 Hz) นาน 500 ms

# สถานะการทำงานของกล้อง
camera_active = False
hand_raised = False

running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:  # ปิดโปรแกรม
            running = False
        elif event.type == pygame.MOUSEBUTTONDOWN:  # ตรวจจับการคลิกเมาส์
            # ตรวจสอบว่าคลิกที่ปุ่มเปิด/ปิดกล้อง
            if button_rect.collidepoint(event.pos):
                camera_active = not camera_active  # เปลี่ยนสถานะกล้อง

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
                    if not hand_raised:  # เริ่มเล่นเสียงเมื่อยกแขนครั้งแรก
                        hand_raised = True
                        threading.Thread(target=play_beep, daemon=True).start()
                else:
                    hand_raised = False  # หยุดเสียงเมื่อเอามือลง

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
