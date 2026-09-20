import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

import cv2
import sys
import time
import threading
from recognition_manager import RecognitionManager
from ui_manager import UIManager
from config import CAMERA_INDEX, FRAME_WIDTH, FRAME_HEIGHT


def input_name_threaded(manager: RecognitionManager, result_holder: dict):
    try:
        print("\n" + "=" * 50)
        print("  NEW FACE REGISTRATION")
        print("=" * 50)
        first_name = input("  First name: ").strip()
        last_name = input("  Last name:  ").strip()

        if first_name and last_name:
            person_id = manager.complete_registration(first_name, last_name)
            result_holder["success"] = True
            result_holder["person_id"] = person_id
            print(f"  ✓ Registered as {first_name} {last_name} (ID: {person_id})")
        else:
            print("  ✗ Registration cancelled (empty name)")
            manager.cancel_registration()
            result_holder["success"] = False
    except Exception as e:
        print(f"  ✗ Registration error: {e}")
        manager.cancel_registration()
        result_holder["success"] = False

    result_holder["done"] = True
    print("=" * 50 + "\n")


def print_database(manager: RecognitionManager):
    persons = manager.db.get_all_persons()
    print("\n" + "=" * 70)
    print(f"  DATABASE ({len(persons)} persons)")
    print("=" * 70)
    if not persons:
        print("  (empty)")
    for p in persons:
        emb_count = manager.db.get_embedding_count_for_person(p["id"])
        print(
            f"  ID:{p['id']:3d}  {p['first_name']:15s} {p['last_name']:15s}  "
            f"Seen:{p['recognition_count']:4d}x  "
            f"Embeddings:{emb_count}  "
            f"Last:{p['last_seen_at']}"
        )
    print("=" * 70 + "\n")


def main():
    print("Initializing Face Recognition App...")

    manager = RecognitionManager()
    ui = UIManager()

    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        print(f"ERROR: Cannot open camera {CAMERA_INDEX}")
        sys.exit(1)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

    print("\nControls: [R]egister  [S]kip  [D]atabase  [Q]uit\n")

    reg_thread = None
    reg_result = {"done": False, "success": False}

    frame_count = 0
    cached_results = []

    # FPS measurement
    prev_time = time.time()
    fps = 0.0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1

            # Check if registration thread finished
            if reg_thread is not None and reg_result.get("done"):
                reg_thread.join()
                reg_thread = None
                reg_result = {"done": False, "success": False}

            # PROCESS INFERENCE EVERY 2nd FRAME (Massive speed boost)
            if frame_count % 2 == 0 or manager.registration_in_progress:
                cached_results = manager.process_frame(frame)

            stats = manager.get_stats()

            # Measure real FPS
            curr_time = time.time()
            fps = 0.9 * fps + 0.1 * (1.0 / max(curr_time - prev_time, 1e-5))
            prev_time = curr_time

            # Draw results
            if manager.registration_in_progress:
                frame = ui.draw_registration_prompt(frame)
            else:
                frame = ui.draw_results(frame, cached_results, stats)

            # Display FPS in upper right
            cv2.putText(
                frame, f"FPS: {fps:.1f}", (frame.shape[1] - 130, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2
            )

            cv2.imshow("Face Recognition", frame)

            key = cv2.waitKey(1) & 0xFF

            if key == ord("q") or key == 27:
                break

            elif key == ord("r") or key == ord("R"):
                if not manager.registration_in_progress:
                    pending = None
                    for result in cached_results:
                        if result.get("status") == "pending":
                            pending = result.get("tracker")
                            break

                    if pending:
                        manager.start_registration(pending)
                        reg_result = {"done": False, "success": False}
                        reg_thread = threading.Thread(
                            target=input_name_threaded,
                            args=(manager, reg_result),
                            daemon=True,
                        )
                        reg_thread.start()
                    else:
                        print("[Info] No confirmed unknown face to register. Wait for progress bar to complete.")

            elif key == ord("s") or key == ord("S"):
                if manager.registration_in_progress:
                    manager.cancel_registration()
                    print("[Info] Registration cancelled")
                else:
                    manager._unknown_trackers.clear()
                    print("[Info] Cleared unknown face trackers")

            elif key == ord("d") or key == ord("D"):
                print_database(manager)

    except KeyboardInterrupt:
        print("\nInterrupted by user")

    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("App closed.")


if __name__ == "__main__":
    main()