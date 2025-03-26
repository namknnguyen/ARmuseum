import cv2
import numpy as np
import threading
import warnings
import torch
from transformers import pipeline
import google.generativeai as genai
from elevenlabs import play
from elevenlabs.client import ElevenLabs
import sounddevice as sd
from scipy.io.wavfile import write as writeaudio
warnings.filterwarnings("ignore")

#-----------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Initialize all APIs and models

# STT initialization
transcriber = pipeline(model="openai/whisper-base", device=0 if torch.cuda.is_available() else -1)
audio_file_path = "recorded_audio.wav"

# LLM initialization
genai.configure(api_key="")
model = genai.GenerativeModel(
    model_name="gemini-1.5-flash",
    generation_config={
        "temperature": 1,
        "top_p": 0.95,
        "top_k": 64,
        "max_output_tokens": 8192,
        "response_mime_type": "text/plain",
    }
)
chat_session = model.start_chat(history=[])

# TTS initialization
client = ElevenLabs(api_key='')

# Global variables
stop_threads = False
painting_id = None
painting_dict = ["Thiếu nữ bên hoa huệ by Tô Ngọc Vân", "Hai thiếu nữ và em bé by Tô Ngọc Vân"]
painting_lock = threading.Lock()

#-----------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Core functions

def detect_markers(frame, detector):
    """Detect ArUco markers in frame and return corners and IDs"""
    global painting_id
    corners, ids, _ = detector.detectMarkers(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
    frame_copy = frame.copy()
    
    if ids is not None:
        cv2.aruco.drawDetectedMarkers(frame_copy, corners, ids)
        marker_dict = {id[0]: corner for id, corner in zip(ids, corners)}
        
        # First painting (markers 1-4)
        if all(i in marker_dict for i in [1, 2, 3, 4]):
            src_pts = np.array([marker_dict[i][0][j] for i, j in zip([1,2,3,4], [0,1,2,3])], dtype=np.float32)
            with painting_lock:
                painting_id = 0
            return frame_copy, src_pts, 0
            
        # Second painting (markers 5-8)
        if all(i in marker_dict for i in [5, 6, 7, 8]):
            src_pts = np.array([marker_dict[i][0][j] for i, j in zip([5,6,7,8], [0,1,2,3])], dtype=np.float32)
            with painting_lock:
                painting_id = 1
            return frame_copy, src_pts, 1
            
    return frame_copy, None, None

def overlay_video(frame, video_frame, src_pts, video_dims):
    """Overlay video frame onto source frame using perspective transform"""
    dst_pts = np.array([[video_dims[0]-1, 0],
                        [0, 0], 
                        [0, video_dims[1]-1],
                        [video_dims[0]-1, video_dims[1]-1]], 
                        dtype=np.float32)

    warped = cv2.warpPerspective(video_frame, 
                               cv2.getPerspectiveTransform(dst_pts, src_pts),
                               (frame.shape[1], frame.shape[0]))

    mask = np.zeros_like(frame)
    cv2.fillConvexPoly(mask, src_pts.astype(int), (255,255,255))
    frame = cv2.bitwise_and(frame, cv2.bitwise_not(mask))
    frame = cv2.bitwise_or(frame, cv2.bitwise_and(warped, mask))
    return frame

def record_audio():
    while not stop_threads:
        print("Press Enter to start recording.")
        input() 
        print("Recording... Press Enter to stop.")
        audio_data = sd.rec(int(1e6), samplerate=48000, channels=1, dtype='int16') 
        input()  
        sd.stop() 
        transcription = transcribe(audio_data)
        print(transcription)
        answer = chat(transcription)
        print(answer)
        readaloudEL(answer)

def transcribe(audio_data):
    writeaudio(audio_file_path, 44100, audio_data)
    return transcriber(audio_file_path)['text']

def chat(user_input):
    with painting_lock:
        current_painting = painting_dict[painting_id] if painting_id is not None else "No painting detected"
    prompt = f"Painting: {current_painting}. Prompt: {user_input}. Respond in 50 words or less."
    return chat_session.send_message(prompt).text

def readaloudEL(text_input):
    play(client.generate(
        text=text_input,
        voice="Charlotte",
        model="eleven_multilingual_v2"
    ))

#-----------------------------------------------------------------------------------------------------------------------------------------------------------------------
def main():
    global stop_threads
    
    # Initialize ArUco detector
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
    detector = cv2.aruco.ArucoDetector(aruco_dict, cv2.aruco.DetectorParameters())

    # Initialize video sources
    video1 = cv2.VideoCapture("video.mp4")
    video2 = cv2.VideoCapture("video2.mp4")
    cap = cv2.VideoCapture(0)
    if not (video1.isOpened() and video2.isOpened() and cap.isOpened()):
        print("Error: Could not open video sources")
        exit()

    video1_dims = (int(video1.get(cv2.CAP_PROP_FRAME_WIDTH)), 
                 int(video1.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    video2_dims = (int(video2.get(cv2.CAP_PROP_FRAME_WIDTH)), 
                 int(video2.get(cv2.CAP_PROP_FRAME_HEIGHT)))

    # Create and start recording thread
    threading.Thread(target=record_audio, daemon=True).start()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Detect markers and overlay video
        frame_copy, src_pts, detected_painting = detect_markers(frame, detector)
        
        if src_pts is not None:
            if detected_painting == 0:
                video = video1
                dims = video1_dims
            else:
                video = video2
                dims = video2_dims
                
            ret, video_frame = video.read()
            if not ret:
                video.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ret, video_frame = video.read()
                
            frame_copy = overlay_video(frame_copy, video_frame, src_pts, dims)

        # Display current painting name
        with painting_lock:
            if painting_id is not None:
                cv2.putText(frame_copy, f"Painting: {painting_dict[painting_id]}", (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        cv2.imshow("AR Painting", frame_copy)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            stop_threads = True
            break

    cap.release()
    video1.release()
    video2.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
