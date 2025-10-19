import mediapipe as mp
import cv2
import numpy as np
import pygame

# Initialize pygame mixer for sound playback
pygame.mixer.init()

# Load the sound file using pygame
sound_file = 'C:/Users/nilesh/Desktop/project/ding.wav'
try:
    sound = pygame.mixer.Sound(sound_file)
except pygame.error as e:
    print(f"Error loading sound: {e}")
    sound = None


def play_sound():
    if sound:
        sound.play()


def findAngle(a, b, c, minVis=0.8):
    # Finds the angle at b with endpoints a and c
    # Returns -1 if below minimum visibility threshold
    if a.visibility > minVis and b.visibility > minVis and c.visibility > minVis:
        bc = np.array([c.x - b.x, c.y - b.y, c.z - b.z])
        ba = np.array([a.x - b.x, a.y - b.y, a.z - b.z])

        angle = np.arccos((np.dot(ba, bc)) / (np.linalg.norm(ba) * np.linalg.norm(bc))) * (180 / np.pi)

        if angle > 180:
            return 360 - angle
        else:
            return angle
    else:
        return -1


def legState(angle):
    if angle < 0:
        return 0  # Joint is not being picked up
    elif angle < 105:
        return 1  # Squat range
    elif angle < 150:
        return 2  # Transition range
    else:
        return 3  # Upright range


if __name__ == "_main_":
    # Init mediapipe drawing and pose
    mp_drawing = mp.solutions.drawing_utils
    mp_pose = mp.solutions.pose

    # Initialize device camera (0 is usually the default camera)
    cap = cv2.VideoCapture(0)

    # Check if the camera is accessible
    if not cap.isOpened():
        print("Error: Unable to access the camera")
        exit()

    # Main Detection Loop
    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        # Initialize Reps and Body State
        repCount = 0
        lastState = 9

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                print('Error: Could not load image from camera.')
                break

            frame = cv2.resize(frame, (1024, 600))

            # Convert frame to RGB
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame.flags.writeable = False

            # Detect Pose Landmarks
            results = pose.process(frame)
            lm = results.pose_landmarks
            if lm:
                lm_arr = lm.landmark
            else:
                print("Please Step Into Frame")
                # Convert back to BGR for display
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                cv2.imshow("Squat Rep Counter", frame)
                cv2.waitKey(1)
                continue

            # Allow write, convert back to BGR
            frame.flags.writeable = True
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

            # Draw pose landmarks
            mp_drawing.draw_landmarks(frame, lm, mp_pose.POSE_CONNECTIONS)

            # Calculate Angles for both legs
            rAngle = findAngle(lm_arr[24], lm_arr[26], lm_arr[28])
            lAngle = findAngle(lm_arr[23], lm_arr[25], lm_arr[27])

            # Calculate leg states
            rState = legState(rAngle)
            lState = legState(lAngle)
            state = rState * lState

            # State logic for reps counting
            if state == 0:  # One or both legs not detected
                if rState == 0:
                    print("Right Leg Not Detected")
                if lState == 0:
                    print("Left Leg Not Detected")
            elif state % 2 == 0 or rState != lState:  # One or both legs transitioning
                if lastState == 1:
                    if lState == 2 or lState == 1:
                        print("Fully extend left leg")
                    if rState == 2 or rState == 1:
                        print("Fully extend right leg")
                else:
                    if lState == 2 or lState == 3:
                        print("Fully retract left leg")
                    if rState == 2 or rState == 3:
                        print("Fully retract right leg")
            else:
                if state == 1 or state == 9:
                    if lastState != state:
                        lastState = state
                        if lastState == 1:
                            print("GOOD!")
                            play_sound()  # Play sound when a squat rep is completed
                            repCount += 1

            # Display rep count on frame
            cv2.putText(frame, f"Squats: {repCount}", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            # Display the frame
            cv2.imshow("Squat Rep Counter", frame)
            if cv2.waitKey(1) & 0xFF == 27:  # Press ESC to exit
                break

    cap.release()
    cv2.destroyAllWindows()
