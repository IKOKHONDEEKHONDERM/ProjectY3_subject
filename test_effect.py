import cv2
import numpy as np
import mediapipe as mp
import pygame
import random

# กำหนด Mediapipe
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

# เริ่ม Mediapipe Pose
pose = mp_pose.Pose()

# เริ่มต้น Pygame
pygame.init()
screen_width, screen_height = 640, 480
screen = pygame.display.set_mode((screen_width, screen_height))
pygame.display.set_caption("Pose Detection with Firework Effect")

# กำหนดสี
white = (255, 255, 255)
black = (0, 0, 0)
colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255), (0, 255, 255)]

snowflakes = []

# เปิดกล้อง
cap = cv2.VideoCapture(0)


def draw_fog(surface, density):
    for _ in range(density):
        x = random.randint(0, screen_width)
        y = random.randint(0, screen_height)
        radius = random.randint(30, 50)
        color = (200, 200, 200, random.randint(50, 100))  # สีหมอกโปร่งแสง
        fog_surface = pygame.Surface((radius * 2, radius * 2), pygame.SRCALPHA)
        pygame.draw.circle(fog_surface, color, (radius, radius), radius)
        surface.blit(fog_surface, (x - radius, y - radius))



def draw_snowfall(surface, num_snowflakes):
    global snowflakes
    if len(snowflakes) < num_snowflakes:
        for _ in range(num_snowflakes - len(snowflakes)):
            snowflakes.append([random.randint(0, screen_width), random.randint(0, screen_height), random.randint(2, 5)])

    for flake in snowflakes:
        flake[1] += random.randint(1, 3)  # ความเร็วของหิมะ
        pygame.draw.circle(surface, white, (flake[0], flake[1]), flake[2])
        if flake[1] > screen_height:
            flake[1] = 0  # เริ่มต้นใหม่จากด้านบน

# ฟังก์ชันสร้างพลุ
def draw_firework(surface, x, y):
    for _ in range(50):  # สุ่มจุดไฟพลุ
        radius = random.randint(2, 5)
        angle = random.uniform(0, 2 * np.pi)
        distance = random.randint(10, 50)
        end_x = int(x + distance * np.cos(angle))
        end_y = int(y + distance * np.sin(angle))
        pygame.draw.circle(surface, random.choice(colors), (end_x, end_y), radius)

def draw_explosion_ring(surface, x, y):
    for radius in range(10, 100, 10):  # วงแหวนขนาดต่าง ๆ
        color = random.choice(colors)
        pygame.draw.circle(surface, color, (x, y), radius, 2)

def draw_rainbow_arc(surface, x, y):
    for i in range(7):  # สีสายรุ้ง
        pygame.draw.arc(
            surface, 
            colors[i % len(colors)], 
            [x - 50 - i * 10, y - 50 - i * 10, 100 + i * 20, 100 + i * 20], 
            np.pi, 
            2 * np.pi, 
            5
        )


# ตัวแปรควบคุมการทำงาน
running = True
firework_triggered = False
firework_position = (0, 0)

while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    # อ่านภาพจากกล้อง
    ret, frame = cap.read()
    if not ret:
        break

    # ประมวลผลภาพ
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(image_rgb)

    # แสดงภาพบนหน้าจอ
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame = np.rot90(frame)
    frame_surface = pygame.surfarray.make_surface(frame)
    screen.blit(frame_surface, (0, 0))

    # ตรวจจับการยกแขน
    if results.pose_landmarks:
        landmarks = results.pose_landmarks.landmark
        left_wrist = landmarks[mp_pose.PoseLandmark.LEFT_WRIST]
        right_wrist = landmarks[mp_pose.PoseLandmark.RIGHT_WRIST]
        left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER]
        right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER]

        # ตรวจสอบว่ามือทั้งสองข้างอยู่สูงกว่าหัวไหล่
        if left_wrist.y < left_shoulder.y and right_wrist.y < right_shoulder.y:
            firework_triggered = True
            firework_position = (screen_width // 2, screen_height // 2)  # กำหนดตำแหน่งพลุ

    # แสดงพลุเมื่อมีการยกแขน
    # if firework_triggered:
    #     draw_firework(screen, *firework_position)
    #     firework_triggered = False  # แสดงพลุครั้งเดียว

    # if firework_triggered:
    #     draw_explosion_ring(screen, screen_width // 2, screen_height // 2)

    draw_snowfall(screen, 50)  # จำนวนเกล็ดหิมะ

    # draw_fog(screen, 50)  # ความหนาแน่นของหมอก


    # if firework_triggered:
    #     draw_rainbow_arc(screen, screen_width // 2, screen_height // 2)



    pygame.display.flip()

cap.release()
pygame.quit()
