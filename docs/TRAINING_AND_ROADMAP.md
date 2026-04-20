# 🧠 HGVCS — Model Training Guide & Advanced Features Roadmap

> A comprehensive guide to improving gesture recognition accuracy through custom ML training and extending HGVCS with next-generation features.

---

## 📌 Table of Contents

1. [Why Train a Custom Model?](#why-train)
2. [How MediaPipe Works (Under the Hood)](#how-mediapipe-works)
3. [Training Pipeline Overview](#pipeline)
4. [Step 1 — Data Collection](#step1)
5. [Step 2 — Data Preprocessing](#step2)
6. [Step 3 — Model Architecture](#step3)
7. [Step 4 — Training](#step4)
8. [Step 5 — Evaluation & Tuning](#step5)
9. [Step 6 — Integrating into HGVCS](#step6)
10. [Advanced Features Roadmap](#advanced-features)

---

## 1. Why Train a Custom Model? {#why-train}

The current `hand_engine.py` uses **rule-based classification** — hardcoded finger-state comparisons. This works for simple gestures but has serious limits:

| Problem | Impact |
|---|---|
| Fixed thresholds | Fails for small/large hands, different skin tones |
| No learning | Cannot adapt to your specific hand shape |
| Brittle to lighting | Landmark positions shift under bad light |
| Can't learn new gestures | Must write new code for every new sign |
| No temporal context | Can't distinguish "hold" from "quick flash" |

A **trained ML classifier** on top of MediaPipe landmarks solves all of these — accuracy jumps from ~70% to **95%+** on custom gestures.

---

## 2. How MediaPipe Works (Under the Hood) {#how-mediapipe-works}

```
Camera Frame (BGR)
      |
      v
MediaPipe HandLandmarker
      |
      v
21 Landmarks x (x, y, z)  <-- 63 raw numbers per hand
      |
      v
Your Classifier  <--  THIS is what you train
      |
      v
Gesture Label + Confidence
```

MediaPipe gives you **21 hand keypoints** already — you never retrain MediaPipe itself. You train a lightweight classifier **on top of those 63 numbers**.

### Landmark Map

```
                 8   12  16  20
                 |   |   |   |
                 7   11  15  19
             4   |   |   |   |
             |   6   10  14  18
             3   |   |   |   |
             |   5---9--13--17
             2   |
         1       |
          \      |
           0 (WRIST)
```

---

## 3. Training Pipeline Overview {#pipeline}

```
|                    TRAINING PIPELINE                        |
|                                                             |
|  1. Collect  ->  2. Preprocess  ->  3. Train  ->  4. Export|
|                                                             |
|  Live camera     Normalize         LSTM /         .h5      |
|  Save landmarks  Augment           MLP /          .tflite  |
|  Label gesture   Split datasets    Transformer    ONNX     |
```

---

## 4. Step 1 — Data Collection {#step1}

### Create the collection script

```python
# scripts/collect_gesture_data.py

import cv2, csv, os, time, mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

GESTURES = [
    "open_palm", "closed_fist", "pointing", "peace_sign",
    "thumbs_up", "thumbs_down", "ok_sign", "rock_on",
    "three_fingers", "four_fingers", "swipe_left", "swipe_right",
    "grab_hold", "wave", "phone_sign", "pinch",
    # Add custom gestures below
    "my_custom_gesture",
]

SAMPLES_PER_GESTURE = 200
OUTPUT_CSV = "data/training/gesture_dataset.csv"

def extract_landmarks(detection_result):
    if not detection_result.hand_landmarks:
        return None
    hand  = detection_result.hand_landmarks[0]
    wrist = hand[0]
    row   = []
    for lm in hand:
        row.extend([lm.x - wrist.x, lm.y - wrist.y, lm.z - wrist.z])
    return row   # 63 wrist-normalized values

def collect(gesture_name, samples=SAMPLES_PER_GESTURE):
    model_path = "models/hand_landmarker.task"
    base_opts  = mp_python.BaseOptions(model_asset_path=model_path)
    opts = mp_vision.HandLandmarkerOptions(
        base_options=base_opts,
        running_mode=mp_vision.RunningMode.VIDEO,
        num_hands=1,
        min_hand_detection_confidence=0.6,
    )
    landmarker = mp_vision.HandLandmarker.create_from_options(opts)
    cap   = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    count = 0; ts_ms = 0; rows = []; recording = False

    print(f"\n[COLLECT] Gesture: '{gesture_name}'")
    print("  Press SPACE to start recording, Q to quit.\n")

    while count < samples:
        ret, frame = cap.read()
        if not ret: break
        frame  = cv2.flip(frame, 1)
        rgb    = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        ts_ms += 33
        det = landmarker.detect_for_video(mp_img, ts_ms)
        lm  = extract_landmarks(det)

        status = f"Recording: {count}/{samples}" if recording else "Press SPACE to start"
        color  = (0, 220, 100) if recording else (80, 80, 200)
        cv2.putText(frame, f"Gesture: {gesture_name}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, status, (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        cv2.imshow("HGVCS Data Collector", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord(' '): recording = True
        elif key == ord('q'): break

        if recording and lm is not None:
            rows.append([gesture_name] + lm)
            count += 1
            time.sleep(0.05)

    cap.release(); landmarker.close(); cv2.destroyAllWindows()

    mode = 'a' if os.path.exists(OUTPUT_CSV) else 'w'
    with open(OUTPUT_CSV, mode, newline='') as f:
        writer = csv.writer(f)
        if mode == 'w':
            writer.writerow(['gesture'] + [f'x{i}' for i in range(63)])
        writer.writerows(rows)
    print(f"  Saved {count} samples for '{gesture_name}'")

if __name__ == "__main__":
    import sys
    target   = sys.argv[1] if len(sys.argv) > 1 else None
    gestures = [target] if target else GESTURES
    for g in gestures:
        collect(g)
```

### Run it

```bash
# Collect ALL gestures (~3 seconds per gesture × 16 = ~1 minute)
python scripts/collect_gesture_data.py

# Collect a single gesture
python scripts/collect_gesture_data.py thumbs_up
```

> [!TIP]
> **Variety is the single most important factor.** Change lighting, distance, and hand angle every 50 samples. This alone can add 8-12% accuracy.

---

## 5. Step 2 — Data Preprocessing {#step2}

```python
# scripts/preprocess_data.py

import pandas as pd, numpy as np, pickle, os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

CSV_PATH = "data/training/gesture_dataset.csv"
OUT_DIR  = "data/training/processed"

def augment(X):
    """4x data augmentation on landmark arrays."""
    return np.vstack([
        X,                                           # original
        X + np.random.normal(0, 0.01, X.shape),     # Gaussian noise
        X * 0.92,                                    # scale down
        X * 1.08,                                    # scale up
    ])

def preprocess():
    os.makedirs(OUT_DIR, exist_ok=True)
    df = pd.read_csv(CSV_PATH)
    print(f"Dataset: {len(df)} rows, {df['gesture'].nunique()} classes")

    X = df.drop('gesture', axis=1).values.astype('float32')
    y = df['gesture'].values
    le = LabelEncoder()
    y_enc = le.fit_transform(y)

    X_aug = augment(X)
    y_aug = np.tile(y_enc, 4)

    X_tr, X_te, y_tr, y_te = train_test_split(
        X_aug, y_aug, test_size=0.15, stratify=y_aug, random_state=42)
    X_tr, X_va, y_tr, y_va = train_test_split(
        X_tr, y_tr, test_size=0.12, stratify=y_tr, random_state=42)

    for name, arr in [('X_train',X_tr),('X_val',X_va),('X_test',X_te),
                      ('y_train',y_tr),('y_val',y_va),('y_test',y_te)]:
        np.save(f"{OUT_DIR}/{name}.npy", arr)
    with open(f"{OUT_DIR}/label_encoder.pkl", 'wb') as f:
        pickle.dump(le, f)

    print(f"Train: {len(X_tr)}  Val: {len(X_va)}  Test: {len(X_te)}")
    print(f"Classes: {list(le.classes_)}")

if __name__ == "__main__":
    preprocess()
```

---

## 6. Step 3 — Model Architecture {#step3}

### Option A — MLP (Fast, ~94% accuracy on static gestures)

```python
import tensorflow as tf
from tensorflow.keras import layers, regularizers

def build_mlp(n_classes):
    return tf.keras.Sequential([
        layers.Input(shape=(63,)),
        layers.BatchNormalization(),
        layers.Dense(256, activation='relu',
                     kernel_regularizer=regularizers.l2(1e-4)),
        layers.Dropout(0.35),
        layers.Dense(128, activation='relu',
                     kernel_regularizer=regularizers.l2(1e-4)),
        layers.Dropout(0.25),
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.15),
        layers.Dense(n_classes, activation='softmax'),
    ], name="gesture_mlp")
```

### Option B — LSTM (Best for swipes, circular, wave motion)

```python
def build_lstm(n_classes, seq_len=20):
    inp = layers.Input(shape=(seq_len, 63))
    x   = layers.LSTM(128, return_sequences=True)(inp)
    x   = layers.Dropout(0.3)(x)
    x   = layers.LSTM(64)(x)
    x   = layers.Dropout(0.2)(x)
    x   = layers.Dense(64, activation='relu')(x)
    out = layers.Dense(n_classes, activation='softmax')(x)
    return tf.keras.Model(inp, out, name="gesture_lstm")
```

### Option C — Hybrid (Recommended — best of both)

```python
def build_hybrid(n_classes, seq_len=20):
    # Static branch (single frame)
    static_in = layers.Input(shape=(63,), name="static")
    s = layers.Dense(128, activation='relu')(static_in)
    s = layers.BatchNormalization()(s)
    s = layers.Dropout(0.3)(s)
    s = layers.Dense(64, activation='relu')(s)

    # Temporal branch (frame sequence)
    seq_in = layers.Input(shape=(seq_len, 63), name="sequence")
    t = layers.LSTM(64)(seq_in)
    t = layers.Dropout(0.2)(t)

    merged = layers.Concatenate()([s, t])
    out    = layers.Dense(64, activation='relu')(merged)
    out    = layers.Dense(n_classes, activation='softmax')(out)
    return tf.keras.Model([static_in, seq_in], out, name="gesture_hybrid")
```

---

## 7. Step 4 — Training {#step4}

```python
# scripts/train_gesture_model.py

import numpy as np, pickle, tensorflow as tf
from tensorflow.keras.callbacks import (
    EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, TensorBoard
)

DATA_DIR    = "data/training/processed"
MODEL_PATH  = "models/gesture_classifier.h5"
TFLITE_PATH = "models/gesture_classifier.tflite"

def train():
    X_train = np.load(f"{DATA_DIR}/X_train.npy")
    X_val   = np.load(f"{DATA_DIR}/X_val.npy")
    X_test  = np.load(f"{DATA_DIR}/X_test.npy")
    y_train = np.load(f"{DATA_DIR}/y_train.npy")
    y_val   = np.load(f"{DATA_DIR}/y_val.npy")
    y_test  = np.load(f"{DATA_DIR}/y_test.npy")
    with open(f"{DATA_DIR}/label_encoder.pkl", 'rb') as f:
        le = pickle.load(f)

    n_classes = len(le.classes_)
    model = build_mlp(n_classes)   # swap to build_lstm or build_hybrid
    model.summary()

    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    callbacks = [
        EarlyStopping('val_accuracy', patience=15, restore_best_weights=True),
        ReduceLROnPlateau('val_loss', factor=0.5, patience=7, min_lr=1e-6),
        ModelCheckpoint(MODEL_PATH, monitor='val_accuracy', save_best_only=True),
        TensorBoard(log_dir='logs/tensorboard'),
    ]

    model.fit(X_train, y_train, validation_data=(X_val, y_val),
              epochs=150, batch_size=64, callbacks=callbacks, verbose=1)

    loss, acc = model.evaluate(X_test, y_test, verbose=0)
    print(f"\n  Test Accuracy: {acc*100:.2f}%")

    # Export to TFLite for fast inference
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    with open(TFLITE_PATH, 'wb') as f:
        f.write(converter.convert())

    print(f"  Saved: {MODEL_PATH}")
    print(f"  Saved: {TFLITE_PATH}")

if __name__ == "__main__":
    train()
```

### Full workflow

```bash
python scripts/collect_gesture_data.py   # ~1 minute at camera
python scripts/preprocess_data.py        # < 5 seconds
python scripts/train_gesture_model.py    # 2-5 minutes on CPU
tensorboard --logdir logs/tensorboard    # visualise curves in browser
```

---

## 8. Step 5 — Evaluation & Tuning {#step5}

### Confusion Matrix

```python
import numpy as np, pickle, tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report

model = tf.keras.models.load_model("models/gesture_classifier.h5")
with open("data/training/processed/label_encoder.pkl", 'rb') as f:
    le = pickle.load(f)
X_test = np.load("data/training/processed/X_test.npy")
y_test = np.load("data/training/processed/y_test.npy")

y_pred = model.predict(X_test).argmax(axis=1)
cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(14, 12))
sns.heatmap(cm, annot=True, fmt='d', xticklabels=le.classes_,
            yticklabels=le.classes_, cmap='Blues')
plt.title("Gesture Confusion Matrix"); plt.tight_layout()
plt.savefig("docs/confusion_matrix.png", dpi=150)
print(classification_report(y_test, y_pred, target_names=le.classes_))
```

### Accuracy targets

| Metric | Acceptable | Good | Excellent |
|---|---|---|---|
| Overall accuracy | > 85% | > 92% | > 97% |
| Per-class accuracy | > 80% | > 90% | > 95% |
| Inference time | < 30 ms | < 15 ms | < 5 ms |
| False positive rate | < 10% | < 5% | < 2% |

### Common problems and fixes

| Problem | Fix |
|---|---|
| Low accuracy on one gesture | Collect 100 more samples of that gesture |
| Underfitting (low train acc) | Larger model, more epochs |
| Overfitting (val acc drops) | More dropout, more augmentation |
| Two gestures confused | Collect more samples at different angles |
| Slow inference | Use TFLite model instead of .h5 |

---

## 9. Step 6 — Integrate ML into HGVCS {#step6}

Add this class to `hand_engine.py`, then swap in `process()`:

```python
import tensorflow as tf, pickle, numpy as np

class MLGestureClassifier:
    """Drop-in replacement for the rule-based classify_gesture()."""

    def __init__(self,
                 model_path="models/gesture_classifier.tflite",
                 encoder_path="data/training/processed/label_encoder.pkl"):
        self._interp = tf.lite.Interpreter(model_path=model_path)
        self._interp.allocate_tensors()
        self._in_idx  = self._interp.get_input_details()[0]['index']
        self._out_idx = self._interp.get_output_details()[0]['index']
        with open(encoder_path, 'rb') as f:
            self._le = pickle.load(f)
        self._threshold = 0.70

    def predict(self, landmarks):
        wrist = landmarks[0]
        row = np.array(
            [v for lm in landmarks
             for v in [lm.x-wrist.x, lm.y-wrist.y, lm.z-wrist.z]],
            dtype='float32'
        ).reshape(1, 63)
        self._interp.set_tensor(self._in_idx, row)
        self._interp.invoke()
        probs = self._interp.get_tensor(self._out_idx)[0]
        idx   = int(probs.argmax())
        conf  = float(probs[idx])
        if conf < self._threshold:
            return "unknown", conf
        return self._le.classes_[idx], conf
```

In `HandEngine.__init__`:
```python
try:
    self._ml = MLGestureClassifier()
    self._use_ml = True
except Exception as e:
    self._use_ml = False
    print(f"[HandEngine] Rule-based mode (ML not loaded): {e}")
```

In `HandEngine.process()`:
```python
# Replace the classify_gesture() call with:
if self._use_ml:
    gesture, conf = self._ml.predict(hand)
else:
    gesture, conf = classify_gesture(hand, self._prev_pinch, ...)
```

---

## 10. Advanced Features Roadmap {#advanced-features}

---

### Feature 1 — Two-Hand Combined Gestures
**Impact: HIGH** | Doubles gesture vocabulary

Detect gestures that require both hands simultaneously:
- `Left fist + Right open palm` → Lock and Send
- `Both hands spread apart` → Maximize window
- `Left pointing + Right thumbs up` → Approve hovered item

**How:** Track `hand_landmarks[0]` and `hand_landmarks[1]`, classify each independently, then use a pair lookup table.

---

### Feature 2 — Per-User Personalized Model
**Impact: HIGH** | +10% accuracy with 3 minutes of your data

Collect 30 samples per gesture per user and fine-tune only the last 2 layers:

```python
for layer in model.layers[:-3]:
    layer.trainable = False
model.compile(optimizer=Adam(1e-4), ...)
model.fit(user_X, user_y, epochs=20)
model.save(f"models/users/{username}.h5")
```

Load user profile at startup. Each person gets their own tuned model.

---

### Feature 3 — Gesture Macro Sequences
**Impact: MEDIUM** | Enables complex command shortcuts

Define gesture sequences that trigger macros:
- `thumbs_up → peace_sign → open_palm` → Open browser
- `closed_fist × 2 (double tap)` → Select all
- `swipe_right × 3 rapid` → Fast-forward media

**How:** Sliding window of last 5 gestures → hash lookup table. No ML needed, zero added latency.

---

### Feature 4 — True Voice + Gesture Fusion
**Impact: HIGH** | Professional-grade accuracy

Combine hand gesture and voice command simultaneously in a 2-second window:
- `Pointing + "Open"` → Open hoverd item
- `Grab Hold + "Send to John"` → Send file to John specifically
- `Three fingers + "Spotify"` → Volume up in Spotify only

The `InputFusionEngine` stub is already in the codebase — implement it with a time-windowed event buffer.

---

### Feature 5 — Adaptive Online Learning
**Impact: VERY HIGH** | System improves while you use it

When HGVCS recognises a gesture wrongly, user can press a correction key:

```python
feedback.add(landmarks, correct_label="thumbs_up")
# Every 100 corrections -> background retrain (10 minutes)
if feedback.count % 100 == 0:
    threading.Thread(target=retrain_background).start()
```

Accuracy compounds over weeks without any manual effort.

---

### Feature 6 — Eye Gaze Integration
**Impact: MEDIUM** | Precision targeting

Use MediaPipe Face Landmarker iris tracking to know what the user is looking at, then act on that target:
- `Gaze at file + Grab Hold` → Select that exact file
- `Gaze at button + Pinch` → Click that button precisely
- `Gaze at paragraph + Thumbs Up` → Highlight paragraph

---

### Feature 7 — Sign Language Mode (ASL A-Z)
**Impact: MEDIUM** | Full accessibility feature

Train on the Kaggle ASL alphabet dataset (80,000 images, free) to recognise letters A-Z and digits 0-9. Add a sequence decoder for word composition:

```
B → R → O → W → S → E → R  (held sequence)  →  launches browser
```

Enables complete text input using only hand signs.

---

### Feature 8 — Mobile Companion App
**Impact: HIGH** | HGVCS anywhere

A Flutter or React Native app that:
- Mirrors the camera feed on phone
- Lets phone be a second gesture input device
- Remotely controls desktop HGVCS over LAN WebSocket

The WebSocket server is already stubbed in `NetworkManager`.

---

### Feature 9 — P2P Remote Desktop Control
**Impact: VERY HIGH** | Killer collaborative feature

Share screen with a LAN peer, and let them send gesture commands back:
- Peer A shows screen to Peer B
- Peer B makes pointing gesture
- Peer A's cursor moves, controlled remotely by gesture

**Stack:** LAN WebSocket (already present) + compressed JPEG stream + PyAutoGUI on the receiving end.

---

### Feature 10 — Game Controller Mode
**Impact: MEDIUM** | Gaming use case

Map gestures to virtual gamepad using `vgamepad`:

| Gesture | Gamepad action |
|---|---|
| Open palm | Jump (A button) |
| Pointing | Aim (right stick) |
| Pinch | Shoot (RT trigger) |
| Rock On | Special move |
| Wave | Pause menu |
| Swipe | D-pad direction |

---

### Feature 11 — Usage Analytics Dashboard
**Impact: LOW-MEDIUM** | Understand your patterns

Track and visualise:
- Most-used gestures per hour/day
- Miss-recognition rate over time
- Average confidence trend
- Action response latency histogram

Built with Plotly Dash served locally on `localhost:8050`.

---

### Feature 12 — Cloud Peer Relay (Internet File Share)
**Impact: HIGH** | Beyond LAN

When sender and receiver are on different networks:

```
Sender ----> HGVCS Relay Server ----> Receiver
            (WebRTC TURN relay)
```

End-to-end encrypted with user public keys (`cryptography` already in requirements.txt).

---

## Recommended Implementation Order

| Timeline | Feature | Effort | Impact |
|---|---|---|---|
| Week 1-2 | Train MLP model on your own data | Low | Very High |
| Week 2 | Personalized user calibration | Low | High |
| Week 3 | Two-hand combined gestures | Medium | High |
| Week 4 | Gesture macro sequences | Low | Medium |
| Week 5-6 | Voice + Gesture fusion | Medium | High |
| Week 7 | Analytics dashboard | Low | Medium |
| Week 8+ | Eye gaze / Sign language | High | Medium |
| Week 10+ | Remote desktop / Cloud relay | High | Very High |

---

## Additional Packages Needed

```bash
# Training
pip install scikit-learn seaborn matplotlib tensorboard

# Virtual gamepad
pip install vgamepad

# Analytics dashboard
pip install plotly dash

# Cloud relay
pip install aiortc

# Sign language dataset
pip install kaggle
```

---

> [!IMPORTANT]
> **Minimum data:** 200 samples per gesture across varied lighting. At 20 FPS collection that is only **10 seconds of recording per gesture** — very fast to build a good dataset.

> [!TIP]
> **Start simple:** Train the MLP first. It runs in under 2 minutes, needs no GPU, and immediately boosts accuracy to ~93%. Move to Hybrid LSTM only if swipe/circular gestures are still misclassified.

> [!NOTE]
> You never retrain MediaPipe itself. You only train a small lightweight classifier that processes the 63 landmark numbers MediaPipe already gives you. This is why training is so fast and works on any laptop.
