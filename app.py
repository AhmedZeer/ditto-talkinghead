# app.py
import streamlit as st
import cv2
from stream_pipeline_online import StreamSDK
import threading
import queue
import numpy as np
import tempfile
import os
import librosa
import math
import pickle
import time

# Function to load pickled configurations
def load_pkl(pkl_path):
    with open(pkl_path, "rb") as f:
        return pickle.load(f)

# Function to run the StreamSDK
def run_sdk(SDK, audio, setup_kwargs, run_kwargs):
    # Setup SDK
    SDK.setup(setup_kwargs['source_path'], setup_kwargs['output_path'], **setup_kwargs)
    
    # Setup Nd
    fade_in = run_kwargs.get("fade_in", -1)
    fade_out = run_kwargs.get("fade_out", -1)
    ctrl_info = run_kwargs.get("ctrl_info", {})
    SDK.setup_Nd(N_d=setup_kwargs['num_frames'], fade_in=fade_in, fade_out=fade_out, ctrl_info=ctrl_info)
    
    online_mode = SDK.online_mode
    if online_mode:
        chunksize = run_kwargs.get("chunksize", (3, 5, 2))
        audio = np.concatenate([np.zeros((chunksize[0] * 640,), dtype=np.float32), audio], 0)
        split_len = int(sum(chunksize) * 0.04 * 16000) + 80  # 6480
        for i in range(0, len(audio), chunksize[1] * 640):
            audio_chunk = audio[i:i + split_len]
            if len(audio_chunk) < split_len:
                audio_chunk = np.pad(audio_chunk, (0, split_len - len(audio_chunk)), mode="constant")
            SDK.run_chunk(audio_chunk, chunksize)
    else:
        aud_feat = SDK.wav2feat.wav2feat(audio)
        SDK.audio2motion_queue.put(aud_feat)
    
    SDK.close()

# Streamlit App
def main():
    st.title("Real-Time Avatar Motion Generation")
    st.write("Upload an audio file and a source image/video to generate a motion video in real-time.")

    # Sidebar for configurations
    st.sidebar.header("Configurations")
    cfg_pkl = st.sidebar.text_input("Config Pickle Path", value="./checkpoints/ditto_cfg/v0.4_hubert_cfg_trt.pkl")
    # data_root = st.sidebar.text_input("Data Root Path", value="./checkpoints/ditto_trt_Ampere_Plus") # For ampere
    data_root = st.sidebar.text_input("Data Root Path", value="./checkpoints/ditto_trt")

    # File uploads
    audio_file = st.file_uploader("Upload Audio (.wav)", type=["wav"])
    source_file = st.file_uploader("Upload Source Image/Video", type=["png", "jpg", "jpeg", "mp4"])

    # Fade in/out configurations
    st.sidebar.subheader("Fade In/Out Settings")
    fade_in = st.sidebar.number_input("Fade In Frames", min_value=-1, value=-1)
    fade_out = st.sidebar.number_input("Fade Out Frames", min_value=-1, value=-1)

    # Control information
    st.sidebar.subheader("Control Information")
    ctrl_info_input = st.sidebar.text_area("Control Info (pickle path or dictionary)", value="")

    # Start button
    if st.button("Start Processing"):
        if audio_file is None or source_file is None:
            st.error("Please upload both audio and source files.")
            return

        # Load configurations
        if os.path.exists(cfg_pkl):
            config = load_pkl(cfg_pkl)
        else:
            st.error(f"Config pickle not found at {cfg_pkl}")
            return

        # Parse control info
        if ctrl_info_input.endswith(".pkl") and os.path.exists(ctrl_info_input):
            ctrl_info = load_pkl(ctrl_info_input)
        else:
            try:
                ctrl_info = eval(ctrl_info_input) if ctrl_info_input else {}
            except:
                ctrl_info = {}
        
        # Save uploaded files to temporary files
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_audio:
            temp_audio.write(audio_file.read())
            temp_audio_path = temp_audio.name
        
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(source_file.name)[1]) as temp_source:
            temp_source.write(source_file.read())
            temp_source_path = temp_source.name

        # Define output path
        temp_output = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        temp_output_path = temp_output.name
        temp_output.close()

        # Load audio using librosa
        audio, sr = librosa.core.load(temp_audio_path, sr=16000)
        num_f = math.ceil(len(audio) / 16000 * 25)  # Assuming 25 FPS

        # Prepare setup and run kwargs
        setup_kwargs = {
            "source_path": temp_source_path,
            "output_path": temp_output_path,
            "num_frames": num_f
        }

        run_kwargs = {
            "fade_in": fade_in,
            "fade_out": fade_out,
            "ctrl_info": ctrl_info,
            "chunksize": (3, 5, 2)  # Example chunksize
        }

        # Initialize display queue
        display_queue = queue.Queue(maxsize=100)

        # Initialize StreamSDK with display_queue
        SDK = StreamSDK(cfg_pkl=cfg_pkl, data_root=data_root, display_queue=display_queue)

        # Start the SDK processing in a separate thread
        processing_thread = threading.Thread(target=run_sdk, args=(SDK, audio, setup_kwargs, run_kwargs))
        processing_thread.start()

        st.success("Processing started!")

        # Placeholder for video frames
        frame_placeholder = st.empty()

        # Display frames as they come
        while processing_thread.is_alive() or not display_queue.empty():
            try:
                frame = display_queue.get(timeout=1)
                # Convert frame to displayable format if needed
                # Assuming frame is in RGB format as a NumPy array
                frame_placeholder.image(frame, channels="RGB", use_column_width=True)
            except queue.Empty:
                continue

        st.success("Processing completed!")

        # Clean up temporary files
        os.remove(temp_audio_path)
        os.remove(temp_source_path)
        os.remove(temp_output_path)

        # Optionally, provide the output video for download
        # with open(temp_output_path, "rb") as video_file:
        #     video_bytes = video_file.read()
        #     st.video(video_bytes)

if __name__ == "__main__":
    main()
