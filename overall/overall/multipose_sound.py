import tensorflow as tf
import tensorflow_hub as hub
import cv2
import pygame
import numpy as np

# โหลด MoveNet Module
model = hub.load('https://tfhub.dev/google/movenet/multipose/lightning/1')
movenet = model.signatures['serving_default']

# เริ่มต้น pygame mixer และกำหนดจำนวน Channel
pygame.mixer.init()
pygame.mixer.set_num_channels(5)  # จำกัดจำนวน Channels สูงสุดให้ 5

# ฟังก์ชันเล่นเสียง
def play_sound_limited(file):
    """เล่นเสียงใน Channel ที่ว่าง หรือหยุด Channel แรกถ้าเต็ม"""
    for channel_id in range(5):  # ลูปตรวจสอบทั้ง 5 Channels
        channel = pygame.mixer.Channel(channel_id)
        if not channel.get_busy():  # ถ้าช่องว่าง
            sound = pygame.mixer.Sound(file)  # โหลดเสียง
            channel.play(sound)  # เล่นเสียงใน Channel นี้
            return
    # ถ้าไม่มีช่องว่าง ให้หยุดช่องแรกและเล่นเสียงใหม่
    channel = pygame.mixer.Channel(0)
    channel.stop()
    sound = pygame.mixer.Sound(file)
    channel.play(sound)

# ฟังก์ชันตรวจจับท่าทางที่ต้องการ (ตัวอย่างเช่น การยกมือ)
def is_pose_detected(keypoints, confidence_threshold):
    """กำหนดเงื่อนไขสำหรับท่าทาง (เช่น การยกมือ หรือยืนในรูปแบบที่เฉพาะ)"""
    left_wrist = keypoints[9]  # ข้อมือซ้าย
    left_elbow = keypoints[7]  # ศอกซ้าย
    right_wrist = keypoints[10]  # ข้อมือขวา
    right_elbow = keypoints[8]  # ศอกขวา
    hip = keypoints[11]  # สะโพก
    left_ankle = keypoints[15]  # ข้อเท้าซ้าย
    right_ankle = keypoints[16]  # ข้อเท้าขวา
    left_knee = keypoints[13]  # เข่าซ้าย
    right_knee = keypoints[14]  # เข่าขวา

    lw_y, lw_x, lw_conf = left_wrist
    le_y, le_x, le_conf = left_elbow
    rw_y, rw_x, rw_conf = right_wrist
    re_y, re_x, re_conf = right_elbow
    hip_y, hip_x, hip_conf = hip
    la_y, la_x, la_conf = left_ankle
    ra_y, ra_x, ra_conf = right_ankle
    lk_y, lk_x, lk_conf = left_knee
    rk_y, rk_x, rk_conf = right_knee

    # ตรวจจับท่าทาง: ข้อมือซ้ายอยู่เหนือศอกซ้าย
    if lw_conf > confidence_threshold and le_conf > confidence_threshold:
        if lw_y < le_y:
            return "left_hand_up"  # ท่าการยกมือซ้าย

    # ตรวจจับท่าทาง: ข้อมือขวาอยู่เหนือศอกขวา
    if rw_conf > confidence_threshold and re_conf > confidence_threshold:
        if rw_y < re_y:
            return "right_hand_up"  # ท่าการยกมือขวา

    # ตรวจจับท่าทาง: ยืนตรง
    if hip_conf > confidence_threshold:
        if abs(hip_y - re_y) < 50 and abs(hip_y - le_y) < 50:
            return "standing"  # ท่ายืนตรง

    # ตรวจจับท่าทาง: การนั่ง
    if hip_conf > confidence_threshold:
        if hip_y > la_y and hip_y > ra_y:  # ถ้าสะโพกต่ำกว่าข้อเท้า
            return "sit"  # ท่านั่ง

    # ตรวจจับท่าทาง: มือซ้ายแตะเข่าซ้าย และมือขวาแตะเข่าขวา
    if lw_conf > confidence_threshold and rw_conf > confidence_threshold and lk_conf > confidence_threshold and rk_conf > confidence_threshold:
        if abs(lw_y - lk_y) < 100 and abs(lw_x - lk_x) < 100 and abs(rw_y - rk_y) < 100 and abs(rw_x - rk_x) < 100:
            return "both_hands_knee"  # ท่ามือทั้งสองข้างแตะเข่าทั้งสองข้าง
        
    # ตรวจจับท่ายกมือซ้ายและขวา
    if lw_conf > confidence_threshold and rw_conf > confidence_threshold:
        if lw_y < le_y and rw_y < re_y:  # ข้อมือซ้ายและขวาอยู่เหนือศอก
            return "both_hands_up"  # ท่ายกมือทั้งสองข้าง

    return None  # ถ้าไม่พบท่าทางที่กำหนด

# ฟังก์ชันวาด Keypoints
def draw_keypoints(frame, keypoints, confidence_threshold):
    y, x, _ = frame.shape
    shaped = np.squeeze(np.multiply(keypoints, [y, x, 1]))
    
    for kp in shaped:
        ky, kx, kp_conf = kp
        if kp_conf > confidence_threshold:
            cv2.circle(frame, (int(kx), int(ky)), 6, (0, 255, 0), -1)

# Dictionary สำหรับกำหนดเส้นเชื่อมต่อ
EDGES = {
    (0, 1): 'm',
    (0, 2): 'c',
    (1, 3): 'm',
    (2, 4): 'c',
    (0, 5): 'm',
    (0, 6): 'c',
    (5, 7): 'm',
    (7, 9): 'm',
    (6, 8): 'c',
    (8, 10): 'c',
    (5, 6): 'y',
    (5, 11): 'm',
    (6, 12): 'c',
    (11, 12): 'y',
    (11, 13): 'm',
    (13, 15): 'm',
    (12, 14): 'c',
    (14, 16): 'c'
}

# ฟังก์ชันวาด Connections
def draw_connections(frame, keypoints, edges, confidence_threshold):
    y, x, _ = frame.shape
    shaped = np.squeeze(np.multiply(keypoints, [y, x, 1]))
    
    for edge, color in edges.items():
        p1, p2 = edge
        y1, x1, c1 = shaped[p1]
        y2, x2, c2 = shaped[p2]
        
        if (c1 > confidence_threshold) & (c2 > confidence_threshold):      
            cv2.line(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 4)

# ฟังก์ชันวนลูปตรวจจับทุกคนในเฟรม
def loop_through_people(frame, keypoints_with_scores, edges, confidence_threshold):
    for person in keypoints_with_scores:
        draw_connections(frame, person, edges, confidence_threshold)
        draw_keypoints(frame, person, confidence_threshold)

        # ตรวจจับท่าทางและเล่นเสียงเมื่อพบ
        pose = is_pose_detected(person, confidence_threshold)
        if pose == "left_hand_up":
            play_sound_limited("D:/WORK/Year-3-2/design/overall/overall/used_sound_file/DI_Dead Kick_Press_0115.6.wav")  # ท่าการยกมือซ้าย
        elif pose == "right_hand_up":
            play_sound_limited('D:/WORK/Year-3-2/design/overall/overall/used_sound_file/DI_HiHat_Foot_1144.4.wav')  # ท่าการยกมือขวา
        elif pose == "standing":
            play_sound_limited('D:/WORK/Year-3-2/design/overall/overall/used_sound_file/Overhead Sample 4.wav')  # ท่ายืนตรง
        elif pose == "sit":
            play_sound_limited('D:/WORK/Year-3-2/design/overall/overall/used_sound_file/Snare Sample 27.wav')  # ท่านั่ง
        elif pose == "both_hands_knee":
            play_sound_limited('D:/WORK/Year-3-2/design/overall/overall/used_sound_file/Tom Sample 17.wav')  # ท่ามือทั้งสองข้างแตะเข่าทั้งสองข้าง

# เริ่มต้น Webcam
cap = cv2.VideoCapture(0)
while cap.isOpened():
    ret, frame = cap.read()
    
    # Resize image สำหรับโมเดล
    img = frame.copy()
    img = tf.image.resize_with_pad(tf.expand_dims(img, axis=0), 192, 256)
    input_img = tf.cast(img, dtype=tf.int32)

    # ตรวจจับ keypoints
    results = movenet(input_img)
    keypoints_with_scores = results['output_0'].numpy()[:, :, :51].reshape((6, 17, 3))
    
    # วาดผลลัพธ์ลงบน frame
    loop_through_people(frame, keypoints_with_scores, EDGES, confidence_threshold=0.3)

    for person in keypoints_with_scores:
        pose = is_pose_detected(person, confidence_threshold=0.3)
        if pose == "both_hands_up":  # ท่ายกมือสองข้าง
            print("Detected: Both hands raised! Exiting...")
            cap.release()
            cv2.destroyAllWindows()
            exit()  # หยุดการทำงานของโปรแกรม

    cv2.imshow('MoveNet Multipose', frame)
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
