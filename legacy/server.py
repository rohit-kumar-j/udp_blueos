# pi_server.py
# This script should be run on your Raspberry Pi (sukote@raspberrypi)

import socket
import threading
import sys
import time
import numpy as np
import pickle
import os
import signal
from select import select
from PIL import Image
import io # Import io for BytesIO

# --- Camera-specific import (using OpenCV now) ---
import cv2

# --- Configuration ---
PI_IP = '192.168.2.10' # Raspberry Pi's static eth0 IP
FEDORA_IP = '192.168.2.1' # Fedora machine's static Ethernet IP

VIDEO_SEND_PORT = 65430        # Server sends video to Client on this port
CLIENT_FLOAT_RECV_PORT = 65431 # Server receives floats from Client on this port
SERVER_FLOAT_SEND_PORT = 65432 # Server sends floats to Client on this port

BUFFER_SIZE = 65536 # Max UDP packet size. Adjust if your encoded frames exceed this. Max for IPv4 is ~65507 bytes.

# --- Global Flags for Thread Control ---
RUNNING = True

def generate_float_array():
    """Generates a dummy NumPy array of floats."""
    return np.random.rand(10, 10).astype(np.float32) # 10x10 array of float32

def send_video_stream():
    """
    Thread function to capture and send actual video frames from the camera
    to the client using OpenCV's VideoCapture.
    """
    global RUNNING
    cap = None
    try:
        print("[SERVER:VideoSender] Initializing camera with OpenCV...")
        # Use 0 for the default camera. If you have multiple, you might need to try 1, 2, etc.
        cap = cv2.VideoCapture(0) 
        
        if not cap.isOpened():
            raise IOError("Cannot open webcam with OpenCV")

        # Set resolution (optional, might not be supported by all cameras or drivers)
        # Match client display resolution if possible for consistency
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        # Set desired FPS (optional, actual FPS might vary)
        cap.set(cv2.CAP_PROP_FPS, 20) 

        print(f"[SERVER:VideoSender] Camera opened. Resolution: {int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x{int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}")

        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.bind((PI_IP, 0)) # Bind to the specific interface IP, let OS pick ephemeral port
        print(f"[SERVER:VideoSender] UDP socket bound to {sock.getsockname()}")

        frame_count = 0
        while RUNNING:
            ret, frame = cap.read() # ret is a boolean (True if frame is read correctly), frame is the image
            if not ret:
                print("[SERVER:VideoSender] Failed to grab frame, retrying...")
                time.sleep(0.1) # Small delay before trying again
                continue

            try:
                # Encode the frame (NumPy array) to JPEG bytes
                # .jpg is the file extension, 90 is the quality (0-100)
                # This directly creates bytes from the OpenCV frame, no PIL conversion needed
                encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 75] # Adjust quality (0-100)
                _, frame_bytes = cv2.imencode('.jpg', frame, encode_param)
                frame_bytes = frame_bytes.tobytes()

                # Check if the frame size exceeds the buffer size
                if len(frame_bytes) > BUFFER_SIZE:
                    print(f"[SERVER:VideoSender] Warning: Frame size ({len(frame_bytes)} bytes) exceeds BUFFER_SIZE ({BUFFER_SIZE} bytes). Frame might be truncated or lost.")
                    # You might need to lower quality, resolution, or increase BUFFER_SIZE (up to UDP max ~65507)

                # Send the encoded frame to the client's video receiving port
                sock.sendto(frame_bytes, (FEDORA_IP, VIDEO_SEND_PORT))
                frame_count += 1
                # print(f"[SERVER:VideoSender] Sent video frame {frame_count} ({len(frame_bytes)} bytes)")
                
                # Adjust sleep time to control frame rate and network load
                # Aiming for ~20 FPS if camera supports it and network allows
                time.sleep(0.05) 

            except Exception as e:
                print(f"[SERVER:VideoSender] Error capturing or sending frame: {e}")
                time.sleep(0.1) # Wait a bit before retrying after an error

    except IOError as e:
        print(f"[SERVER:VideoSender] Camera access error: {e}. Ensure camera is connected, enabled, and not in use by another application.")
        RUNNING = False # Stop other threads if camera fails
    except Exception as e:
        print(f"[SERVER:VideoSender] Critical thread error: {e}")
        RUNNING = False # Stop other threads if camera fails
    finally:
        if cap and cap.isOpened():
            print("[SERVER:VideoSender] Releasing camera...")
            cap.release() # Release the camera resource
            print("[SERVER:VideoSender] Camera released.")
        if 'sock' in locals() and sock:
            sock.close()
        print("[SERVER:VideoSender] Thread stopped.")


def send_server_float_stream():
    """
    Thread function to send float arrays (NumPy arrays) to the client.
    """
    global RUNNING
    try:
        # Create a UDP socket for sending floats
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
            sock.bind((PI_IP, 0)) # Bind to the specific interface IP
            print(f"[SERVER:ServerFloatSender] Bound to {sock.getsockname()}")

            array_count = 0
            while RUNNING:
                float_array = generate_float_array()
                try:
                    # Pickle the NumPy array to bytes for transmission
                    serialized_array = pickle.dumps(float_array)
                    # Send the serialized array to the client's float receiving port
                    sock.sendto(serialized_array, (FEDORA_IP, SERVER_FLOAT_SEND_PORT))
                    array_count += 1
                    # print(f"[SERVER:ServerFloatSender] Sent float array {array_count} ({len(serialized_array)} bytes)")
                except Exception as e:
                    print(f"[SERVER:ServerFloatSender] Error sending float array: {e}")
                time.sleep(0.5) # Send array every 500ms
    except Exception as e:
        print(f"[SERVER:ServerFloatSender] Thread error: {e}")
    finally:
        print("[SERVER:ServerFloatSender] Thread stopped.")


def receive_client_float_stream():
    """
    Thread function to receive float arrays from the client.
    Uses select for non-blocking receive.
    """
    global RUNNING
    try:
        # Create a UDP socket for receiving floats from the client
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
            sock.bind((PI_IP, CLIENT_FLOAT_RECV_PORT)) # Bind to the specific interface IP and port
            print(f"[SERVER:ClientFloatReceiver] Listening on {PI_IP}:{CLIENT_FLOAT_RECV_PORT}")

            while RUNNING:
                # Use select to wait for data with a timeout
                # readable: list of sockets ready for reading
                # writable: list of sockets ready for writing (not used for simple UDP recv)
                # exceptional: list of sockets with exceptional conditions
                readable, _, _ = select([sock], [], [], 1.0) # 1.0 second timeout

                if readable:
                    data, addr = sock.recvfrom(BUFFER_SIZE)
                    try:
                        # Unpickle the received bytes back into a NumPy array
                        received_array = pickle.loads(data)
                        print(f"[SERVER:ClientFloatReceiver] Received float array from {addr}:")
                        # print(received_array) # Uncomment to see full array
                        print(f"  Shape: {received_array.shape}, Dtype: {received_array.dtype}")
                    except (pickle.UnpicklingError, ValueError) as e:
                        print(f"[SERVER:ClientFloatReceiver] Error unpickling data from {addr}: {e}")
                    except Exception as e:
                        print(f"[SERVER:ClientFloatReceiver] Unexpected error processing data from {addr}: {e}")
    except Exception as e:
        print(f"[SERVER:ClientFloatReceiver] Thread error: {e}")
    finally:
        print("[SERVER:ClientFloatReceiver] Thread stopped.")


def signal_handler(sig, frame):
    """Handles graceful shutdown on Ctrl+C."""
    global RUNNING
    print("\n[SERVER] Ctrl+C detected. Shutting down gracefully...")
    RUNNING = False
    # Give threads a moment to finish their loops
    time.sleep(1) 
    sys.exit(0)

def main():
    signal.signal(signal.SIGINT, signal_handler)

    print(f"[{os.uname().nodename}] Starting Multi-Stream UDP Server (with OpenCV Camera!)...")

    # Start all communication threads
    video_sender_thread = threading.Thread(target=send_video_stream)
    client_float_receiver_thread = threading.Thread(target=receive_client_float_stream)
    server_float_sender_thread = threading.Thread(target=send_server_float_stream)

    video_sender_thread.start()
    client_float_receiver_thread.start()
    server_float_sender_thread.start()

    # Keep the main thread alive until all other threads finish
    # This is important for graceful shutdown with Ctrl+C
    try:
        while RUNNING:
            time.sleep(0.5)
    except KeyboardInterrupt:
        # This block might not be hit if signal_handler catches first
        pass
    finally:
        print("[SERVER] Main thread exiting. Waiting for worker threads...")
        # Join threads with a timeout, so they don't hang indefinitely if something goes wrong
        video_sender_thread.join(timeout=5)
        client_float_receiver_thread.join(timeout=5)
        server_float_sender_thread.join(timeout=5)
        print("[SERVER] All threads stopped. Server shut down.")

if __name__ == "__main__":
    main()

