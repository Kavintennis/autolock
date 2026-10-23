
import cv2
import mediapipe as mp
import time
import ctypes
import asyncio
import threading
import tkinter as tk
import traceback

from winsdk.windows.media.control import (
    GlobalSystemMediaTransportControlsSessionManager,
    GlobalSystemMediaTransportControlsSessionPlaybackStatus
)

# ============ SETTINGS ============

LOCK_AFTER = 6.0
CONFIRM_AWAY = 0.0

CAMERA_INDEX = 0
FRAME_WIDTH = 640
FRAME_HEIGHT = 480

MEDIA_CHECK_INTERVAL = 1.0

HEAD_MIN = 0.30
HEAD_MAX = 0.70
EYE_MIN = 0.25
EYE_MAX = 0.75

mp_face_mesh = mp.solutions.face_mesh

# Shared state
detection_enabled = threading.Event()
detection_enabled.set()

shutdown_event = threading.Event()


# ============ FOCUS BUTTON ============

def create_focus_button():
    root = tk.Tk()
    root.title("Smart Focus")
    root.attributes("-topmost", True)
    root.overrideredirect(True)
    root.geometry("145x48+30+100")
    root.configure(bg="#202124")

    button = tk.Button(
        root,
        text="FOCUS: ON",
        font=("Segoe UI", 10, "bold"),
        bg="#188038",
        fg="white",
        activebackground="#137333",
        activeforeground="white",
        relief="flat",
        bd=0,
        cursor="hand2",
        command=lambda: toggle_focus(button)
    )
    button.pack(fill="both", expand=True, padx=3, pady=3)

    # Drag the button with the right mouse button.
    drag = {"x": 0, "y": 0}

    def start_drag(event):
        drag["x"] = event.x_root - root.winfo_x()
        drag["y"] = event.y_root - root.winfo_y()

    def move_window(event):
        x = event.x_root - drag["x"]
        y = event.y_root - drag["y"]
        root.geometry(f"+{x}+{y}")

    button.bind("<ButtonPress-3>", start_drag)
    button.bind("<B3-Motion>", move_window)

    def close_app():
        shutdown_event.set()
        detection_enabled.clear()
        root.destroy()

    root.protocol("WM_DELETE_WINDOW", close_app)

    return root, button


def toggle_focus(button):
    if detection_enabled.is_set():
        detection_enabled.clear()
        button.config(
            text="FOCUS: OFF",
            bg="#B3261E",
            activebackground="#8C1D18"
        )
    else:
        detection_enabled.set()
        button.config(
            text="FOCUS: ON",
            bg="#188038",
            activebackground="#137333"
        )


# ============ VIDEO PLAYBACK DETECTION ============

async def get_media_playing():
    try:
        manager = (
            await GlobalSystemMediaTransportControlsSessionManager
            .request_async()
        )

        for session in manager.get_sessions():
            info = session.get_playback_info()

            if (
                info is not None
                and info.playback_status
                == GlobalSystemMediaTransportControlsSessionPlaybackStatus.PLAYING
            ):
                return True

        return False

    except Exception:
        return False


def media_is_playing():
    try:
        return asyncio.run(get_media_playing())
    except Exception:
        return False


# ============ EYE DETECTION ============

def eye_position(landmarks, corner_a, corner_b, iris_ids):
    a = landmarks[corner_a].x
    b = landmarks[corner_b].x

    iris_x = sum(
        landmarks[i].x for i in iris_ids
    ) / len(iris_ids)

    left, right = min(a, b), max(a, b)
    width = right - left

    if width < 0.001:
        return None

    return (iris_x - left) / width


def estimate_attention(landmarks):
    left = landmarks[234]
    right = landmarks[454]
    nose = landmarks[1]

    if abs(right.x - left.x) < 0.001:
        return False

    ratio = (nose.x - left.x) / (right.x - left.x)

    if right.x < left.x:
        ratio = 1.0 - ratio

    head_forward = HEAD_MIN < ratio < HEAD_MAX

    eye_a = eye_position(
        landmarks, 33, 133, range(468, 473)
    )
    eye_b = eye_position(
        landmarks, 362, 263, range(473, 478)
    )

    if eye_a is None or eye_b is None:
        return head_forward

    return (
        head_forward
        and EYE_MIN < eye_a < EYE_MAX
        and EYE_MIN < eye_b < EYE_MAX
    )


# ============ CAMERA CLEANUP ============

def release_resources(camera, face_mesh):
    if camera is not None:
        try:
            camera.release()
        except Exception:
            pass

    if face_mesh is not None:
        try:
            face_mesh.close()
        except Exception:
            pass


# ============ BACKGROUND DETECTION WORKER ============

def detection_worker():
    camera = None
    face_mesh = None

    away_since = None
    candidate_since = None
    has_seen_face = False

    last_media_check = 0.0
    media_playing = False

    while not shutdown_event.is_set():
        try:
            # Focus OFF: turn off the camera and release MediaPipe.
            if not detection_enabled.is_set():
                release_resources(camera, face_mesh)
                camera = None
                face_mesh = None

                away_since = None
                candidate_since = None
                has_seen_face = False

                time.sleep(0.2)
                continue

            # Initialize face detection when Focus turns ON.
            if face_mesh is None:
                face_mesh = mp_face_mesh.FaceMesh(
                    max_num_faces=1,
                    refine_landmarks=True,
                    min_detection_confidence=0.6,
                    min_tracking_confidence=0.6
                )

            # Open camera only when needed.
            if camera is None:
                camera = cv2.VideoCapture(CAMERA_INDEX)

                if not camera.isOpened():
                    camera.release()
                    camera = None
                    time.sleep(2)
                    continue

                camera.set(
                    cv2.CAP_PROP_FRAME_WIDTH,
                    FRAME_WIDTH
                )
                camera.set(
                    cv2.CAP_PROP_FRAME_HEIGHT,
                    FRAME_HEIGHT
                )

                away_since = None
                candidate_since = None
                has_seen_face = False
                last_media_check = 0.0
                media_playing = False

            ok, frame = camera.read()

            if not ok:
                camera.release()
                camera = None
                time.sleep(0.5)
                continue

            if (
                shutdown_event.is_set()
                or not detection_enabled.is_set()
            ):
                continue

            now = time.monotonic()

            # Check Windows media sessions periodically.
            if now - last_media_check >= MEDIA_CHECK_INTERVAL:
                media_playing = media_is_playing()
                last_media_check = now

            # Suspend locking while media is playing.
            if media_playing:
                away_since = None
                candidate_since = None
                continue

            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = face_mesh.process(rgb)

            face_present = bool(result.multi_face_landmarks)
            looking = False

            if face_present:
                landmarks = (
                    result.multi_face_landmarks[0].landmark
                )
                looking = estimate_attention(landmarks)

            if not has_seen_face:
                if face_present:
                    has_seen_face = True
                    away_since = None
                    candidate_since = None

            elif looking:
                away_since = None
                candidate_since = None

            else:
                if candidate_since is None:
                    candidate_since = now

                if now - candidate_since >= CONFIRM_AWAY:
                    if away_since is None:
                        away_since = now

                    if now - away_since >= LOCK_AFTER:
                        # Recheck Focus before locking.
                        if (
                            detection_enabled.is_set()
                            and not shutdown_event.is_set()
                        ):
                            # Check playback again before locking.
                            if media_is_playing():
                                away_since = None
                                candidate_since = None
                                continue

                            release_resources(camera, face_mesh)
                            camera = None
                            face_mesh = None

                            ctypes.windll.user32.LockWorkStation()

                            # Restart the worker after unlock.
                            time.sleep(2)

                        away_since = None
                        candidate_since = None
                        has_seen_face = False

        except Exception:
            # Recover from errors without permanently ending the worker.
            release_resources(camera, face_mesh)
            camera = None
            face_mesh = None

            away_since = None
            candidate_since = None
            has_seen_face = False

            # Brief pause before retrying.
            time.sleep(2)

    release_resources(camera, face_mesh)


# ============ START APPLICATION ============

if __name__ == "__main__":
    worker = threading.Thread(
        target=detection_worker,
        daemon=True
    )
    worker.start()

    root, button = create_focus_button()

    try:
        root.mainloop()
    finally:
        shutdown_event.set()
        detection_enabled.clear()
