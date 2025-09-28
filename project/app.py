from flask import Flask, request, jsonify, render_template, Response
from flask_socketio import SocketIO
import cv2
import base64
import mediapipe as mp
import threading
import numpy as np
import pygame
import requests  # For HTTP communication with laser ESP32
import json
import time

# Initialize Flask app and SocketIO
app = Flask(__name__)
socketio = SocketIO(app)

# --- Laser ESP32 Configuration ---
LASER_ESP32_IP = "172.20.10.5"  # Replace with your laser ESP32's IP
LASER_API_URL = f"http://{LASER_ESP32_IP}"

# --- Exercise Tracking Variables ---
active_processes = {}
video_captures = {}
rep_counts = {}

# --- Laser System State ---
laser_active = False

# --- Audio Setup ---
pygame.mixer.init()
try:
    sound = pygame.mixer.Sound('ding.wav')  # Update path as needed
except pygame.error as e:
    print(f"Error loading sound: {e}")
    sound = None

# --- Utility Functions ---
def play_sound():
    if sound:
        sound.play()

def findAngle(a, b, c, minVis=0.8):
    if a.visibility > minVis and b.visibility > minVis and c.visibility > minVis:
        bc = np.array([c.x - b.x, c.y - b.y, c.z - b.z])
        ba = np.array([a.x - b.x, a.y - b.y, a.z - b.z])
        angle = np.arccos((np.dot(ba, bc)) / (np.linalg.norm(ba) * np.linalg.norm(bc))) * (180 / np.pi)
        return 360 - angle if angle > 180 else angle
    return -1

def legState(angle):
    if angle < 0: return 0
    elif angle < 105: return 1
    elif angle < 150: return 2
    return 3

# --- Video Processing Thread ---
def video_feed_thread(mode):
    try:
        cap = cv2.VideoCapture('http://172.20.10.3:81/stream')
        if not cap.isOpened():
            print("Error: Could not connect to IP camera stream")
            return
            
        video_captures[mode] = cap
        mp_pose = mp.solutions.pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
        mp_drawing = mp.solutions.drawing_utils
        
        rep_count = 0
        last_state = 9
        rep_counts[mode] = 0

        # Skipping-specific variables
        if mode == "skipping":
            jump_threshold = 0.03
            smoothing_window = 5
            y_positions = []
            last_jump_frame = 0
            min_frames_between_jumps = 15

        frame_count = 0

        while True:
            success, frame = cap.read()
            if not success:
                print("Failed to read frame from IP camera")
                cap.release()
                cap = cv2.VideoCapture('http://172.20.10.3:81/stream')
                continue

            frame_count += 1
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = mp_pose.process(frame_rgb)

            if results.pose_landmarks:
                frame = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
                mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp.solutions.pose.POSE_CONNECTIONS)
                lm_arr = results.pose_landmarks.landmark
                
                if mode == "squats":
                    rAngle = findAngle(lm_arr[24], lm_arr[26], lm_arr[28])
                    lAngle = findAngle(lm_arr[23], lm_arr[25], lm_arr[27])
                    rState = legState(rAngle)
                    lState = legState(lAngle)
                    state = rState * lState
                    
                    if state == 1 or state == 9:
                        if last_state != state:
                            last_state = state
                            if last_state == 1:
                                rep_counts[mode] += 1
                                play_sound()
                
                elif mode == "skipping":
                    current_y = (lm_arr[23].y + lm_arr[24].y + lm_arr[25].y + lm_arr[26].y) / 4
                    y_positions.append(current_y)
                    if len(y_positions) > smoothing_window:
                        y_positions.pop(0)
                    
                    if len(y_positions) == smoothing_window:
                        avg_y = sum(y_positions) / len(y_positions)
                        if (current_y < avg_y - jump_threshold and 
                            frame_count - last_jump_frame > min_frames_between_jumps):
                            rep_counts[mode] += 1
                            last_jump_frame = frame_count

            _, buffer = cv2.imencode('.jpg', frame)
            frame_bytes = base64.b64encode(buffer).decode('utf-8')
            socketio.emit('video_feed', {
                'image': frame_bytes,
                'count': rep_counts[mode]
            })

            if mode not in active_processes:
                break

        cap.release()
    except Exception as e:
        print(f"Error in video feed thread: {str(e)}")

# --- Flask Routes ---
@app.route('/')
def home():
    return render_template('eg1.html')

@app.route('/camera')
def camera():
    return render_template('camera.html')

@app.route('/feed')
def feed():
    mode = request.args.get('mode')
    if not mode:
        return jsonify({"error": "Mode parameter is missing."}), 400
    return render_template('feed.html', mode=mode)

@app.route('/laser')
def laser():
    return render_template('laser.html')

@app.route('/start_mode')
def start_mode():
    mode = request.args.get('mode', '').strip()
    if not mode:
        return jsonify({"error": "Mode parameter is missing."}), 400
    
    try:
        if mode.lower() not in ['squats', 'skipping']:
            return jsonify({"error": "Invalid mode"}), 400
        
        if mode in active_processes:
            return jsonify({"message": f"{mode.capitalize()} mode is already running!"}), 200
        
        thread = threading.Thread(target=video_feed_thread, args=(mode,))
        thread.daemon = True
        thread.start()
        active_processes[mode] = thread
        return jsonify({"message": f"{mode.capitalize()} mode started successfully!"}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/stop_mode')
def stop_mode():
    try:
        active_processes.clear()
        for cap in video_captures.values():
            cap.release()
        video_captures.clear()
        rep_counts.clear()
        return jsonify({"message": "All training modes stopped successfully!"}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# --- Laser Control Routes ---
@app.route('/start_laser')
def start_laser():
    global laser_active
    try:
        response = requests.post(f"{LASER_API_URL}/start", timeout=3)
        if response.status_code == 200:
            laser_active = True
            return jsonify({
                "status": "success",
                "message": "Laser activated"
            })
        return jsonify({
            "status": "error",
            "message": "Failed to communicate with laser device"
        }), 500
    except requests.exceptions.RequestException as e:
        return jsonify({
            "status": "error",
            "message": f"Connection error: {str(e)}"
        }), 500
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500

@app.route('/stop_laser')
def stop_laser():
    global laser_active
    try:
        response = requests.post(f"{LASER_API_URL}/stop", timeout=3)
        if response.status_code == 200:
            laser_active = False
            return jsonify({
                "status": "success",
                "message": "Laser deactivated"
            })
        return jsonify({
            "status": "error",
            "message": "Failed to communicate with laser device"
        }), 500
    except requests.exceptions.RequestException as e:
        return jsonify({
            "status": "error",
            "message": f"Connection error: {str(e)}"
        }), 500
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500

@app.route('/laser_status')
def laser_status():
    try:
        response = requests.get(f"{LASER_API_URL}/status", timeout=2)
        if response.status_code == 200:
            return jsonify(response.json())
        return jsonify({
            "status": "error",
            "message": "Could not get laser status"
        }), 500
    except requests.exceptions.RequestException:
        return jsonify({
            "status": "error",
            "message": "Laser device not responding"
        }), 500

if __name__ == '__main__':
    print("🔥 Server running! Open: http://127.0.0.1:5000/ in your browser.")
    socketio.run(app, debug=True, host='0.0.0.0', port=5000)