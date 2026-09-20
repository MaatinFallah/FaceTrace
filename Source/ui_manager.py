"""
UI Manager:
Handles all OpenCV drawing, text rendering, and user interaction overlays.
"""

import cv2
import numpy as np
from config import (
    COLOR_KNOWN,
    COLOR_UNKNOWN,
    COLOR_PENDING,
    COLOR_TEXT_BG,
    COLOR_WHITE,
    FONT_SCALE,
    FONT_THICKNESS,
    UNKNOWN_CONFIRMATION_FRAMES,
)


class UIManager:
    """Draws bounding boxes, labels, and HUD on frames."""

    @staticmethod
    def draw_results(frame: np.ndarray, results: list[dict], stats: dict) -> np.ndarray:
        """
        Draw all face detection results on the frame.
        Returns the annotated frame (modified in-place).
        """
        for result in results:
            bbox = result["bbox"]
            x1, y1, x2, y2 = bbox
            status = result["status"]

            if status == "known":
                UIManager._draw_known_face(frame, result)
            elif status == "pending":
                UIManager._draw_pending_face(frame, result)
            elif status == "registering":
                UIManager._draw_registering_face(frame, result)
            else:  # unknown
                UIManager._draw_unknown_face(frame, result)

        # Draw HUD
        UIManager._draw_hud(frame, stats)

        return frame

    @staticmethod
    def _draw_known_face(frame: np.ndarray, result: dict):
        """Draw bounding box and info for a recognized person."""
        x1, y1, x2, y2 = result["bbox"]
        data = result["person_data"]
        similarity = result.get("similarity", 0)

        # Bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), COLOR_KNOWN, 2)

        # Prepare text lines
        name = f"{data['first_name']} {data['last_name']}"
        sim_text = f"Match: {similarity:.1%}"
        count_text = f"Seen: {data['recognition_count']}x"
        last_seen = f"Last: {data['last_seen_at']}"

        lines = [name, sim_text, count_text, last_seen]
        UIManager._draw_label_box(frame, x1, y1, lines, COLOR_KNOWN)

    @staticmethod
    def _draw_unknown_face(frame: np.ndarray, result: dict):
        """Draw bounding box for an unrecognized face (still tracking)."""
        x1, y1, x2, y2 = result["bbox"]
        frames_seen = result.get("frames_seen", 0)

        cv2.rectangle(frame, (x1, y1), (x2, y2), COLOR_UNKNOWN, 2)

        progress = min(frames_seen / UNKNOWN_CONFIRMATION_FRAMES, 1.0)
        lines = [
            "Unknown Face",
            f"Confirming: {progress:.0%}",
        ]
        UIManager._draw_label_box(frame, x1, y1, lines, COLOR_UNKNOWN)

        # Progress bar below the bounding box
        bar_y = y2 + 5
        bar_w = x2 - x1
        bar_h = 8
        cv2.rectangle(frame, (x1, bar_y), (x2, bar_y + bar_h), (50, 50, 50), -1)
        fill_w = int(bar_w * progress)
        cv2.rectangle(frame, (x1, bar_y), (x1 + fill_w, bar_y + bar_h), COLOR_UNKNOWN, -1)

    @staticmethod
    def _draw_pending_face(frame: np.ndarray, result: dict):
        """Draw bounding box for a confirmed unknown face (ready to register)."""
        x1, y1, x2, y2 = result["bbox"]

        cv2.rectangle(frame, (x1, y1), (x2, y2), COLOR_PENDING, 3)

        lines = [
            "NEW FACE DETECTED",
            "Press 'R' to register",
            "Press 'S' to skip",
        ]
        UIManager._draw_label_box(frame, x1, y1, lines, COLOR_PENDING)

    @staticmethod
    def _draw_registering_face(frame: np.ndarray, result: dict):
        """Draw bounding box while registration is happening."""
        x1, y1, x2, y2 = result["bbox"]

        cv2.rectangle(frame, (x1, y1), (x2, y2), COLOR_PENDING, 3)

        lines = [
            "REGISTERING...",
            "Check console for input",
        ]
        UIManager._draw_label_box(frame, x1, y1, lines, COLOR_PENDING)

    @staticmethod
    def _draw_label_box(
        frame: np.ndarray,
        x: int,
        y: int,
        lines: list[str],
        color: tuple,
    ):
        """
        Draw a semi-transparent background box with text lines above bbox.
        """
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = FONT_SCALE
        thickness = FONT_THICKNESS
        padding = 5
        line_height = 22

        # Calculate box size
        max_width = 0
        for line in lines:
            (w, h), _ = cv2.getTextSize(line, font, scale, thickness)
            max_width = max(max_width, w)

        box_w = max_width + 2 * padding
        box_h = len(lines) * line_height + 2 * padding
        box_y = y - box_h - 5

        # Clamp to frame boundaries
        if box_y < 0:
            box_y = y + (y - box_y) + 10  # draw below instead

        # Semi-transparent background
        overlay = frame.copy()
        cv2.rectangle(
            overlay,
            (x, box_y),
            (x + box_w, box_y + box_h),
            COLOR_TEXT_BG,
            -1,
        )
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

        # Border
        cv2.rectangle(frame, (x, box_y), (x + box_w, box_y + box_h), color, 1)

        # Text
        for i, line in enumerate(lines):
            text_y = box_y + padding + (i + 1) * line_height - 4
            cv2.putText(frame, line, (x + padding, text_y), font, scale, COLOR_WHITE, thickness)

    @staticmethod
    def _draw_hud(frame: np.ndarray, stats: dict):
        """Draw heads-up display with database stats."""
        h, w = frame.shape[:2]
        font = cv2.FONT_HERSHEY_SIMPLEX

        lines = [
            f"DB: {stats['total_persons']} persons, {stats['total_embeddings']} embeddings",
            f"Active: {stats['active_sightings']} tracked, {stats['unknown_trackers']} unknown",
            "Keys: [R]egister  [S]kip  [Q]uit  [D]atabase",
        ]

        y_offset = h - 20
        for line in reversed(lines):
            (tw, th), _ = cv2.getTextSize(line, font, 0.5, 1)
            cv2.rectangle(frame, (5, y_offset - th - 5), (tw + 15, y_offset + 5), COLOR_TEXT_BG, -1)
            cv2.putText(frame, line, (10, y_offset), font, 0.5, COLOR_WHITE, 1)
            y_offset -= 25

    @staticmethod
    def draw_registration_prompt(frame: np.ndarray) -> np.ndarray:
        """Overlay a large registration prompt on the frame."""
        h, w = frame.shape[:2]
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, h), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)

        font = cv2.FONT_HERSHEY_SIMPLEX
        text = "REGISTRATION MODE - Enter name in console"
        (tw, th), _ = cv2.getTextSize(text, font, 1.0, 2)
        x = (w - tw) // 2
        y = h // 2
        cv2.putText(frame, text, (x, y), font, 1.0, COLOR_PENDING, 2)


        return frame