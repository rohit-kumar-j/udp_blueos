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

# --- Configuration ---
PI_IP = '192.168.2.10' # Raspberry Pi's static eth0 IP
FEDORA_IP = '192.168.2.1' # Fedora machine's static Ethernet IP

VIDEO_SEND_PORT = 65430        # Server sends video to Client on this port
CLIENT_FLOAT_RECV_PORT = 65431 # Server receives floats from Client on this port
SERVER_FLOAT_SEND_PORT = 65432 # Server sends floats to Client on this port

BUFFER_SIZE = 65536 # Max UDP packet size (adjust as needed for video frames)

# --- Global Flags for Thread Control ---
RUNNING = True

def generate_dummy_video_frame(width=640, height=480):
    """Generates a dummy grayscale image (video frame) as bytes."""
    # Create a simple grayscale image for demonstration
    # Pixel value based on current time for some variation
    pixel_value = int(time.time() * 10) % 256
    img_array = np.full((height, width), pixel_value, dtype=np.uint8)
    img = Image.fromarray(img_array, 'L') # 'L' for grayscale

    # Convert image to bytes (e.g., JPEG format)
    import io
    byte_arr = io.BytesIO()
    img.save(byte_arr, format='JPEG') # Using JPEG for compression
    return byte_arr.getvalue()

def generate_float_array():
    """Generates a dummy NumPy array of floats."""
    return np.random.rand(10, 10).astype(np.float32) # 10x10 array of float32

def send_video_stream():
    """
    Thread function to send video frames (dummy images) to the client.
    """
    global RUNNING
    try:
        # Create a UDP socket for sending video
        # We bind to the server's IP, let the OS pick an ephemeral port for sending
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
            sock.bind((PI_IP, 0)) # Bind to the specific interface IP
            print(f"[SERVER:VideoSender] Bound to {sock.getsockname()}")

            frame_count = 0
            while RUNNING:
                frame = generate_dummy_video_frame()
                try:
                    # Send the frame to the client's video receiving port
                    sock.sendto(frame, (FEDORA_IP, VIDEO_SEND_PORT))
                    frame_count += 1
                    # print(f"[SERVER:VideoSender] Sent video frame {frame_count} ({len(frame)} bytes)")
                except Exception as e:
                    print(f"[SERVER:VideoSender] Error sending video frame: {e}")
                time.sleep(0.1) # Simulate 10 FPS (100ms delay)
    except Exception as e:
        print(f"[SERVER:VideoSender] Thread error: {e}")
    finally:
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

    print(f"[{os.uname().nodename}] Starting Multi-Stream UDP Server...")

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
        video_sender_thread.join()
        client_float_receiver_thread.join()
        server_float_sender_thread.join()
        print("[SERVER] All threads stopped. Server shut down.")

if __name__ == "__main__":
    main()


