"""
Central configuration for the face recognition app.
All tunable parameters live here.
"""

import os

# --- Paths ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
DB_PATH = os.path.join(DATA_DIR, "faces.db")

# --- Face Detection / Recognition ---
# Cosine similarity threshold: above this = same person
# InsightFace embeddings are normalized, so cosine similarity works well.
# 0.4 is a good balance between precision and recall.
SIMILARITY_THRESHOLD = 0.42

# How many top candidates to consider during ANN search before
# doing exact cosine verification. Higher = more accurate but slower.
ANN_CANDIDATES = 10

# --- Recognition Cooldown ---
# Minimum seconds between counting a "new sighting" for the same person
RECOGNITION_COOLDOWN_SECONDS = 30.0

# --- New Face Confirmation ---
# When an unknown face is detected, how many consecutive frames it must
# appear in before the user is prompted. Prevents phantom detections.
UNKNOWN_CONFIRMATION_FRAMES = 15

# --- Camera ---
CAMERA_INDEX = 0
FRAME_WIDTH = 1280
FRAME_HEIGHT = 720

# --- UI ---
# Colors (BGR)
COLOR_KNOWN = (0, 200, 0)       # Green for recognized
COLOR_UNKNOWN = (0, 0, 255)     # Red for unknown
COLOR_PENDING = (0, 165, 255)   # Orange for pending confirmation
COLOR_TEXT_BG = (0, 0, 0)       # Black background for text
COLOR_WHITE = (255, 255, 255)

# Font
FONT_SCALE = 0.6
FONT_THICKNESS = 2