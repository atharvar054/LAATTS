import cv2
import mediapipe as mp
import numpy as np

class BufferList:
    def _init_(self, buffer_time, default_value=0):
        # Initialize buffer with default values for tracking movement
        self.buffer = [default_value] * buffer_time

    def push(self, value):
        # Add new value and remove oldest one to maintain buffer size
        self.buffer.pop(0)
        self.buffer.append(value)

    def max(self):
        # Get maximum value in buffer
        return max(self.buffer)

    def min(self):
        # Get minimum value in buffer, excluding None values
        return min(filter(lambda x: x is not None, self.buffer), default=0)

    def smooth_update(self, old_value, new_value, alpha=0.5):
        # Apply exponential smoothing to reduce noise
        return alpha * new_value + (1 - alpha) * old_value

def extract_landmarks(results, landmarks_indices, image_width, image_height):
    """Extract specific joint coordinates from pose detection results."""
    return [
        (lm.x * image_width, lm.y * image_height)
        for i, lm in enumerate(results.pose_landmarks.landmark)
        if i in landmarks_indices
    ]

def calculate_center_y(hip_points, shoulder_points):
    """Calculate vertical center and distance between hips and shoulders."""
    cy_hip = int(np.mean([point[1] for point in hip_points]))
    cy_shoulder = int(np.mean([point[1] for point in shoulder_points]))
    return cy_hip, cy_hip - cy_shoulder

def update_counters(cy, cy_shoulder_hip, cy_max, cy_min, flip_flag, thresholds):
    """Update counter logic for detecting skipping motion."""
    dy = cy_max - cy_min
    if dy > thresholds["dy_ratio"] * cy_shoulder_hip:
        # Detect upward movement
        if (cy > cy_max - thresholds["up_ratio"] * dy and 
            flip_flag == thresholds["flag_low"]):
            flip_flag = thresholds["flag_high"]
        # Detect downward movement
        elif (cy < cy_min + thresholds["down_ratio"] * dy and 
              flip_flag == thresholds["flag_high"]):
            flip_flag = thresholds["flag_low"]
    return flip_flag

def draw_visualizations(image, cx, cy, count, image_width, image_height):
    """Draw visual indicators and count on the frame."""
    # Draw centroid point
    cv2.circle(image, (cx, cy), 5, (0, 0, 255), -1)
    cv2.putText(
        image,
        "centroid",
        (cx - 25, cy - 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (0, 0, 255),
        1,
    )
    
    # Draw count
    cv2.putText(
        image,
        f"count = {count}",
        (int(image_width * 0.5), int(image_height * 0.2)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (0, 255, 0),
        1,
    )

def start_detection():
    """Main function to start pose detection and skipping count."""
    # Initialize video capture
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Failed to open camera.")
        return

    # Initialize MediaPipe pose detection
    mp_pose = mp.solutions.pose
    hip_landmarks = [23, 24]
    shoulder_landmarks = [11, 12]

    # Configure detection thresholds
    thresholds = {
        "buffer_time": 50,
        "dy_ratio": 0.3,
        "up_ratio": 0.55,
        "down_ratio": 0.35,
        "flag_low": 150,
        "flag_high": 250,
    }

    # Initialize movement tracking buffers
    buffers = {
        "center_y": BufferList(thresholds["buffer_time"]),
        "center_y_up": BufferList(thresholds["buffer_time"]),
        "center_y_down": BufferList(thresholds["buffer_time"]),
        "center_y_flip": BufferList(thresholds["buffer_time"]),
        "center_y_pref_flip": BufferList(thresholds["buffer_time"]),
    }

    cy_max, cy_min = 100, 100
    flip_flag = thresholds["flag_high"]
    count = 0

    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        while cap.isOpened():
            success, image = cap.read()
            if not success:
                print("Error: Failed to read frame from camera.")
                break

            # Process frame for pose detection
            image_height, image_width, _ = image.shape
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = pose.process(image_rgb)

            if results.pose_landmarks:
                # Extract landmark positions
                hip_points = extract_landmarks(
                    results, hip_landmarks, image_width, image_height
                )
                shoulder_points = extract_landmarks(
                    results, shoulder_landmarks, image_width, image_height
                )
                
                # Calculate centroid position
                cx = int(np.mean([point[0] for point in hip_points]))
                cy, cy_shoulder_hip = calculate_center_y(hip_points, shoulder_points)
            else:
                cx, cy, cy_shoulder_hip = 0, 0, 0

            # Update movement buffers
            buffers["center_y"].push(cy)
            cy_max = buffers["center_y"].smooth_update(cy_max, buffers["center_y"].max())
            buffers["center_y_up"].push(cy_max)
            cy_min = buffers["center_y"].smooth_update(cy_min, buffers["center_y"].min())
            buffers["center_y_down"].push(cy_min)

            # Update skip counting
            prev_flip_flag = flip_flag
            flip_flag = update_counters(
                cy, cy_shoulder_hip, cy_max, cy_min, flip_flag, thresholds
            )
            buffers["center_y_flip"].push(flip_flag)
            buffers["center_y_pref_flip"].push(prev_flip_flag)

            if prev_flip_flag < flip_flag:
                count += 1

            # Draw visualization on frame
            draw_visualizations(image, cx, cy, count, image_width, image_height)

            # Display the processed frame
            cv2.imshow("Skipping Counter", image)

            if cv2.waitKey(5) & 0xFF == 27:  # Exit on ESC
                break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "_main_":
    start_detection()
