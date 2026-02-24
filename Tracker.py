import imutils
import cv2
import numpy as np
import time
import csv
import os



# SETTINGS =====
FRAMES_TO_PERSIST = 5
MIN_SIZE_FOR_MOVEMENT = 5  # smaller for fish
GRID_ROWS = 3
GRID_COLS = 3
VIDEO_SOURCE = "footage.mp4"  # webcam or video file

# CREATE FOLDERS =====
if not os.path.exists("stats"):
    os.makedirs("stats")
if not os.path.exists("recordings"):
    os.makedirs("recordings")

# VIDEO CAPTURE =====
cap = cv2.VideoCapture(VIDEO_SOURCE)

first_frame = None
next_frame = None

# GRID STATS =====
cell_counts = np.zeros((GRID_ROWS, GRID_COLS), dtype=int)

def draw_grid(frame, rows, cols):
    h, w = frame.shape[:2]
    for r in range(1, rows):
        cv2.line(frame, (0, r*h//rows), (w, r*h//rows), (255,255,255), 1)
    for c in range(1, cols):
        cv2.line(frame, (c*w//cols, 0), (c*w//cols, h), (255,255,255), 1)
    return frame

def get_cell(x, y, w, h, frame_width, frame_height, rows, cols):
    # Use the center of the bounding box
    cx = x + w // 2
    cy = y + h // 2
    col = int(cx / frame_width * cols)
    row = int(cy / frame_height * rows)
    return row, col

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = imutils.resize(frame, width=750)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (21, 21), 0)

    if first_frame is None:
        first_frame = gray

    next_frame = gray
    frame_delta = cv2.absdiff(first_frame, next_frame)
    thresh = cv2.threshold(frame_delta, 25, 255, cv2.THRESH_BINARY)[1]
    thresh = cv2.dilate(thresh, None, iterations=2)

    cnts, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    h, w = frame.shape[:2]
    for c in cnts:
        if cv2.contourArea(c) < MIN_SIZE_FOR_MOVEMENT:
            continue
        x, y, cw, ch = cv2.boundingRect(c)
        cv2.rectangle(frame, (x, y), (x+cw, y+ch), (0, 255, 0), 2)
        row, col = get_cell(x, y, cw, ch, w, h, GRID_ROWS, GRID_COLS)
        if 0 <= row < GRID_ROWS and 0 <= col < GRID_COLS:
            cell_counts[row, col] += 1  # increment counter

    frame = draw_grid(frame, GRID_ROWS, GRID_COLS)
    cv2.imshow("Fish Tracker", frame)

    # Update first_frame periodically
    first_frame = next_frame

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# ===== CLEANUP =====
cv2.destroyAllWindows()
cap.release()

# ===== SAVE CSV =====
timestamp = int(time.time())
csv_filename = f"stats/fish_grid_stats_{timestamp}.csv"

with open(csv_filename, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["Row", "Col", "Frames"])
    for r in range(GRID_ROWS):
        for c in range(GRID_COLS):
            writer.writerow([r, c, cell_counts[r, c]])

print(f"Grid visit stats saved to {csv_filename}")
print(cell_counts)
