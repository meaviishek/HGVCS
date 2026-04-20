#!/usr/bin/env python3
"""
Gesture Data Collection Script

This script helps collect training data for gesture recognition.
It captures hand landmarks from the camera and saves them as sequences.

Usage:
    python scripts/collect_gesture_data.py --gesture open_palm --samples 100
"""

import os
import sys
import argparse
import json
import cv2
import numpy as np
import mediapipe as mp
from pathlib import Path
from collections import deque
from datetime import datetime

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

class GestureDataCollector:
    """Collect gesture training data from camera."""
    
    def __init__(self, camera_id=0, sequence_length=30, resolution=(1280, 720)):
        self.camera_id = camera_id
        self.sequence_length = sequence_length
        self.resolution = resolution
        
        # Initialize MediaPipe Hands
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7
        )
        self.mp_draw = mp.solutions.drawing_utils
        
        # Initialize camera
        self.cap = cv2.VideoCapture(camera_id)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, resolution[0])
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, resolution[1])
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        
        # Data buffer
        self.sequence_buffer = deque(maxlen=sequence_length)
        
    def extract_landmarks(self, hand_landmarks):
        """Extract normalized landmarks from MediaPipe result."""
        landmarks = []
        
        # Get wrist position for normalization
        wrist = hand_landmarks.landmark[0]
        
        for lm in hand_landmarks.landmark:
            # Normalize relative to wrist
            landmarks.extend([
                lm.x - wrist.x,
                lm.y - wrist.y,
                lm.z - wrist.z
            ])
        
        # Add finger states (extended = 1, folded = 0)
        finger_tips = [8, 12, 16, 20]
        finger_pips = [6, 10, 14, 18]
        
        for tip, pip in zip(finger_tips, finger_pips):
            tip_y = hand_landmarks.landmark[tip].y
            pip_y = hand_landmarks.landmark[pip].y
            # In image coordinates, lower y means higher in image
            extended = 1 if tip_y < pip_y else 0
            landmarks.append(extended)
        
        return np.array(landmarks)
    
    def collect_gesture(self, gesture_name: str, num_samples: int, output_dir: str):
        """
        Collect samples for a specific gesture.
        
        Args:
            gesture_name: Name of the gesture
            num_samples: Number of samples to collect
            output_dir: Directory to save samples
        """
        output_path = Path(output_dir) / gesture_name
        output_path.mkdir(parents=True, exist_ok=True)
        
        print(f"\nCollecting '{gesture_name}' gesture")
        print(f"Target: {num_samples} samples")
        print(f"Output: {output_path}")
        print("\nInstructions:")
        print("  - Press SPACE to start recording a sample")
        print("  - Hold the gesture steady while recording")
        print("  - Press ESC to cancel current recording")
        print("  - Press 'q' to quit\n")
        
        samples_collected = 0
        is_recording = False
        recording_countdown = 0
        
        while samples_collected < num_samples:
            ret, frame = self.cap.read()
            if not ret:
                print("Error: Failed to capture frame")
                break
            
            # Flip frame for mirror effect
            frame = cv2.flip(frame, 1)
            
            # Process frame
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.hands.process(rgb_frame)
            
            # Create display frame
            display = frame.copy()
            
            # Draw hand landmarks
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    self.mp_draw.draw_landmarks(
                        display,
                        hand_landmarks,
                        self.mp_hands.HAND_CONNECTIONS
                    )
                    
                    # Extract landmarks
                    landmarks = self.extract_landmarks(hand_landmarks)
                    
                    # Add to buffer if recording
                    if is_recording:
                        self.sequence_buffer.append(landmarks)
                        
                        # Check if sequence is complete
                        if len(self.sequence_buffer) >= self.sequence_length:
                            # Save sequence
                            sequence_array = np.array(list(self.sequence_buffer))
                            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                            filename = f"{gesture_name}_{timestamp}.npy"
                            filepath = output_path / filename
                            np.save(filepath, sequence_array)
                            
                            samples_collected += 1
                            is_recording = False
                            self.sequence_buffer.clear()
                            
                            print(f"  Saved: {filename} ({samples_collected}/{num_samples})")
            
            # Draw UI
            self._draw_ui(display, gesture_name, samples_collected, num_samples, is_recording)
            
            # Show frame
            cv2.imshow('Gesture Data Collection', display)
            
            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                print("\nQuitting...")
                break
            
            elif key == 27:  # ESC
                if is_recording:
                    print("  Cancelled recording")
                    is_recording = False
                    self.sequence_buffer.clear()
            
            elif key == 32:  # SPACE
                if not is_recording and results.multi_hand_landmarks:
                    print("  Recording...")
                    is_recording = True
                    self.sequence_buffer.clear()
        
        print(f"\nCompleted! Collected {samples_collected} samples for '{gesture_name}'")
        return samples_collected
    
    def _draw_ui(self, frame, gesture_name, collected, target, is_recording):
        """Draw user interface on frame."""
        h, w = frame.shape[:2]
        
        # Background panel
        cv2.rectangle(frame, (10, 10), (350, 120), (0, 0, 0), -1)
        cv2.rectangle(frame, (10, 10), (350, 120), (255, 255, 255), 2)
        
        # Gesture name
        cv2.putText(frame, f"Gesture: {gesture_name}", (20, 40),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Progress
        progress_text = f"Progress: {collected}/{target}"
        color = (0, 255, 0) if collected >= target else (0, 165, 255)
        cv2.putText(frame, progress_text, (20, 70),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        
        # Recording status
        if is_recording:
            # Blinking red dot
            if int(datetime.now().timestamp() * 2) % 2 == 0:
                cv2.circle(frame, (320, 55), 10, (0, 0, 255), -1)
            cv2.putText(frame, "RECORDING", (220, 100),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        else:
            cv2.putText(frame, "Press SPACE to record", (20, 100),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        # Progress bar
        bar_width = 300
        bar_height = 20
        bar_x = 20
        bar_y = h - 40
        
        # Background
        cv2.rectangle(frame, (bar_x, bar_y), 
                     (bar_x + bar_width, bar_y + bar_height), (50, 50, 50), -1)
        
        # Progress fill
        progress = min(collected / target, 1.0)
        fill_width = int(bar_width * progress)
        cv2.rectangle(frame, (bar_x, bar_y),
                     (bar_x + fill_width, bar_y + bar_height), (0, 255, 0), -1)
        
        # Border
        cv2.rectangle(frame, (bar_x, bar_y),
                     (bar_x + bar_width, bar_y + bar_height), (255, 255, 255), 2)
        
        # Percentage
        percentage = int(progress * 100)
        cv2.putText(frame, f"{percentage}%", (bar_x + bar_width // 2 - 20, bar_y + 15),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    def collect_multiple_gestures(self, gesture_list, samples_per_gesture, output_dir):
        """Collect data for multiple gestures."""
        print("="*60)
        print("HGVCS Gesture Data Collection")
        print("="*60)
        
        total_samples = 0
        for gesture_name in gesture_list:
            collected = self.collect_gesture(gesture_name, samples_per_gesture, output_dir)
            total_samples += collected
            
            if collected < samples_per_gesture:
                print(f"\nWarning: Only collected {collected}/{samples_per_gesture} for '{gesture_name}'")
        
        print("\n" + "="*60)
        print(f"Data collection complete! Total samples: {total_samples}")
        print("="*60)
    
    def close(self):
        """Release resources."""
        self.cap.release()
        cv2.destroyAllWindows()

def main():
    parser = argparse.ArgumentParser(description='Collect gesture training data')
    parser.add_argument('--gesture', type=str,
                        help='Name of gesture to collect (or "all" for all gestures)')
    parser.add_argument('--samples', type=int, default=100,
                        help='Number of samples to collect per gesture')
    parser.add_argument('--output', type=str, default='data/training/gestures',
                        help='Output directory')
    parser.add_argument('--camera', type=int, default=0,
                        help='Camera device ID')
    parser.add_argument('--sequence-length', type=int, default=30,
                        help='Number of frames per sequence')
    parser.add_argument('--config', type=str, default='config/default_config.yaml',
                        help='Config file with gesture list')
    
    args = parser.parse_args()
    
    # Load gesture list from config if available
    gesture_list = []
    if args.gesture == 'all':
        import yaml
        try:
            with open(args.config, 'r') as f:
                config = yaml.safe_load(f)
                gesture_list = list(config['gesture']['mappings'].keys())
        except:
            # Default gesture list
            gesture_list = [
                'open_palm', 'closed_fist', 'thumbs_up', 'thumbs_down',
                'pointing', 'peace_sign', 'three_fingers', 'four_fingers',
                'ok_sign', 'rock_on', 'swipe_left', 'swipe_right',
                'swipe_up', 'swipe_down', 'pinch_in', 'pinch_out'
            ]
    else:
        gesture_list = [args.gesture]
    
    # Create collector
    collector = GestureDataCollector(
        camera_id=args.camera,
        sequence_length=args.sequence_length
    )
    
    try:
        collector.collect_multiple_gestures(
            gesture_list,
            args.samples,
            args.output
        )
    finally:
        collector.close()

if __name__ == "__main__":
    main()
