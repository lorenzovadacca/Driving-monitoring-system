# Driver Monitoring System in Python using MediaPipe: computation of eye and head gaze positions, EAR and PERCLOS (Technologies for Autonomous Vehicles)

A computer vision-based Driver Monitoring System (DMS) for real-time driver distraction, drowsiness, and eye gaze tracking using MediaPipe Face Mesh, Head Pose Estimation (Perspective-n-Point / PnP), Eye Aspect Ratio (EAR) and PERCLOS metrics. Developed for the **Technologies for Autonomous Vehicles** course at Politecnico di Torino.

---

## Table of contents
1. [Overview](#overview)
2. [Requirements](#requirements)
3. [Architecture](#architecture)
   - [Facial Landmark & Iris Detection](#1-facial-landmark--iris-detection-mediapipe-face-mesh)
   - [Head Pose & Gaze Deviation Estimation (SolvePnP)](#2-head-pose--gaze-deviation-estimation-solvepnp)
   - [Eye Aspect Ratio (EAR) & Eye Closure Dynamics](#3-eye-aspect-ratio-ear--eye-closure-dynamics)
   - [Drowsiness Assessment via PERCLOS](#4-drowsiness-assessment-via-perclos)
4. [Specifications](#specifications)
5. [Project structure](#project-structure)
6. [Usage](#usage)
   - [Hardware Setup](#hardware-setup)
   - [Running the System](#running-the-system)
   - [Output Metrics & On-Screen Overlay](#output-metrics--on-screen-overlay)

---

## Overview

Driver Monitoring Systems (DMS) are safety-critical components in SAE Level 2 and Level 3 automated vehicles, guaranteeing driver situational awareness and tracking takeover readiness. 

This project implements a non-intrusive, vision-based real-time DMS that runs on live camera video streams to detect:
* **Gaze & Head Distraction:** Computes head rotation angles ($\text{Pitch}$, $\text{Yaw}$, $\text{Roll}$) using a calibrated pinhole camera model and 3D facial landmarks via `cv2.solvePnP` and Rodrigues rotation decompositions.
* **Prolonged Eye Closure / Microsleep:** Calculates the normalized Eye Aspect Ratio ($\text{EAR}$) from high-resolution eyelid landmarks.
* **Cumulative Drowsiness (PERCLOS):** Evaluates the percentage of eye closure over blink cycles ($t_1 \to t_2 \to t_3 \to t_4$) to raise drowsiness warnings when standard safety thresholds are exceeded.

---

## Requirements

* **Python:** 3.8 to 3.11 recommended.
* **Core Libraries:**
  * `opencv-python` ($\ge 4.5.0$)
  * `mediapipe` ($\ge 0.10.0$)
  * `numpy` ($\ge 1.20.0$)
* **Hardware:** Standard RGB webcam (integrated or USB).

To install the required dependencies:
```bash
pip install opencv-python mediapipe numpy
```

---

## Architecture

```plaintext
[ Webcam Frame (BGR) ] ──> [ RGB Conversion & MediaPipe FaceMesh ]
                                         │
                 ┌───────────────────────┴───────────────────────┐
                 ▼                                               ▼
       [ 468/478 3D Landmarks ]                         [ Iris Centers (468, 473) ]
                 │                                               │
     ┌───────────┴───────────┐                                   │
     ▼                       ▼                                   ▼
[ Head & Eye Pose ]      [ Eye Contours ]                [ 2D Gaze Tracking ]
(SolvePnP + RQDecomp)  (Points 33, 133, 145, 159, ...)   (Overlay on frame)
     │                       │
     ▼                       ▼
[ Pitch, Yaw, Roll ]   [ Eye Aspect Ratio (EAR) ]
     │                       │
     ▼                       ▼
[ Distraction Check ]  [ Normalized EAR ($0 \div 1$) ]
(Combined Deviation)         │
                             ▼
                     [ PERCLOS State Machine ]
                     ($t_1 \to t_2 \to t_3 \to t_4$)
                             │
                             ▼
                     [ Drowsiness Alerts ]
```

### 1. Facial Landmark & Iris Detection (MediaPipe Face Mesh)
* Uses **MediaPipe Face Mesh** with `refine_landmarks=True` to extract 468 standard facial landmarks and additional refined iris landmarks.
* Tracks specific key anatomical points:
  * **Nose tip:** Landmark index `1`
  * **Mouth corners & Chin:** Indices `61`, `291`, `199`
  * **Eye corners & Eyelids:** Right eye (`33`, `133`, `145`, `159`), Left eye (`362`, `263`, `374`, `386`)
  * **Iris centers:** Right eye (`468`), Left eye (`473`)

### 2. Head Pose & Gaze Deviation Estimation (`SolvePnP`)
* **Camera Matrix Formulation:** Configured assuming a standard pinhole camera with optical center at the frame center and focal length proportional to frame width ($f_x = f_y = 1 \times w$, $c_x = w/2$, $c_y = h/2$):
  $$\mathbf{K} = \begin{bmatrix} f & 0 & c_x \\ 0 & f & c_y \\ 0 & 0 & 1 \end{bmatrix}, \quad \mathbf{D} = \mathbf{0}_{4 \times 1}$$
* **Perspective-n-Point Resolution:** Computes rotation ($\mathbf{r}_{\text{vec}}$) and translation ($\mathbf{t}_{\text{vec}}$) vectors via `cv2.solvePnP` mapping 3D generic facial coordinates to 2D image coordinates.
* **Angle Decomposition:** The rotation matrix ($\mathbf{R}$) is obtained using Rodrigues' formula (`cv2.Rodrigues`) and decomposed into Euler angles using `cv2.RQDecomp3x3` to extract $\text{Pitch}$, $\text{Yaw}$, and $\text{Roll}$:
  $$\text{Combined Deviation} = \max\left( |\text{Pitch}|, \, |\text{Yaw}|, \, |\text{Roll}| \right)$$
  When the deviation exceeds the threshold ($30^\circ$), an **ALERT: Driver Distracted** is triggered.

### 3. Eye Aspect Ratio (EAR) & Eye Closure Dynamics
* EAR measures the relative eye opening using vertical and horizontal eyelid Euclidean distances:
  $$\text{EAR} = \frac{|y_{\text{top}} - y_{\text{bottom}}|}{|x_{\text{left}} - x_{\text{right}}|}$$
* **Normalization & Clamping:** To handle individual morphological variations, the average EAR across both eyes is linearly mapped between $0.0$ and $1.0$:
  $$\text{EAR}_{\text{norm}} = \text{clamp}\left( 3.333 \cdot \text{EAR}_{\text{avg}} - 0.5, \, 0.0, \, 1.0 \right)$$
* **Prolonged Closure Detection:** An immediate warning is generated if the normalized EAR remains in an alert state ($\text{EAR}_{\text{norm}} > 0.8$) continuously for more than $10\text{ s}$.

### 4. Drowsiness Assessment via PERCLOS
* Implements the standardized **PERCLOS** (Percentage of Eye Closure) methodology over transient blinking cycles, characterized by four temporal checkpoints:
  * $t_1$: Eye closure onset (transition from $100\%$ open down to $80\%$).
  * $t_2$: Eye actively closing (transition from $80\%$ down to $20\%$).
  * $t_3$: Eye fully closed phase ($20\%$ opening threshold reached).
  * $t_4$: Eye reopening phase (transition from $20\%$ back to $80\%$).
* The cycle PERCLOS value is calculated as:
  $$\text{PERCLOS} = \frac{t_3 - t_2}{t_4 - t_1}$$
* When $\text{PERCLOS} > 0.40$ ($40\%$), the system flags an **ALERT: Drowsiness Detected (PERCLOS)**.

---

## Specifications

| Parameter | Identifier / Symbol | Nominal Threshold | Unit | Description |
| :--- | :---: | :---: | :---: | :--- |
| Distraction Angle Limit | `DISTRACTION_THRESHOLD` | $30.0$ | $\text{deg}$ | Max allowable deviation across Pitch, Yaw, Roll |
| High EAR Baseline | `HIGH_EAR` | $0.80$ | $-$ | Normalized threshold indicating open eye baseline |
| High EAR Alarm Duration | `HIGH_EAR_DURATION` | $10.0$ | $\text{s}$ | Duration trigger for persistent closure warning |
| PERCLOS Alert Threshold | `PERCLOS_ALERT_THRESHOLD` | $0.40$ | $-$ | Minimum ratio ($40\%$) indicating drowsy state |
| EAR Lower Transition Threshold | $-$ | $0.20$ | $-$ | $20\%$ closure threshold for temporal tracking |
| Camera Frame Rate Rate-Limit | `time.sleep(1/25)` | $\sim 25$ | $\text{fps}$ | Nominal internal processing sampling frequency |

---

## Project structure

```plaintext
.
├── SCRIPT-Assignment_Driving_Monitoring_System_with_AI.py   # Main Python real-time DMS application
├── REPORT-Assignment_Driving_Monitoring_System_with_AI.pdf  # Technical project report and validation
└── README.md                                               # Project documentation
```

---

## Usage

### Hardware Setup
1. Connect a functional webcam to your workstation.
2. Ensure the camera has an unobstructed view of the driver's face under reasonable ambient lighting.

### Running the System
1. Open a terminal in the project directory:
   ```bash
   python SCRIPT-Assignment_Driving_Monitoring_System_with_AI.py
   ```
2. The OpenCV GUI window titled `Technologies for Autonomous Vehicles - Driver Monitoring Systems using AI code sample` will launch.
3. To terminate the application, select the video window and press the **`ESC`** key (ASCII `27`).

### Output Metrics & On-Screen Overlay
The live feed displays real-time overlays:
* **Iris Center Coordinates:** Real-time $(x, y)$ coordinate tracking for left (`green`) and right (`red`) pupils.
* **3D Pose Projection Rays:** Directional line vectors indicating head/nose orientation and eye gaze directions.
* **Dynamic Indicators:**
  * `EAR: <value>` (Cyan)
  * `PERCLOS: <value>` (Cyan)
* **Real-Time Visual Alerts (Red):**
  * `ALERT: Driver Distracted (Gaze Deviation)`
  * `ALERT: Drowsiness detected (EAR high)`
  * `ALERT: Drowsiness Detected (PERCLOS)`
