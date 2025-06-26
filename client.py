# fedora_client.py
# This script should be run on your Fedora machine (rohit@fedora)

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
PI_IP = '192.168.2.10'  # Raspberry Pi's static eth0 IP
FEDORA_IP = '192.168.2.1' # Fedora machine's static Ethernet IP

VIDEO_RECV_PORT = 65430        # Client receives video from Server on this port
CLIENT_FLOAT_SEND_PORT = 65431 # Client sends floats to Server on this port
SERVER_FLOAT_RECV_PORT = 65432 # Client receives floats from Server on this port

BUFFER_SIZE = 65536 # Max UDP packet size (adjust as needed for video frames)

# --- Global Flags for Thread Control ---
RUNNING = True

def generate_float_array():
    """Generates a dummy NumPy array of floats."""
    return np.random.rand(5, 5).astype(np.float32) # Smaller array for client to server

def receive_video_stream():
    """
    Thread function to receive video frames from the server.
    Uses select for non-blocking receive.
    """
    global RUNNING
    try:
        # Create a UDP socket for receiving video
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
            sock.bind((FEDORA_IP, VIDEO_RECV_PORT)) # Bind to client's IP and specific port
            print(f"[CLIENT:VideoReceiver] Listening on {FEDORA_IP}:{VIDEO_RECV_PORT}")

            frame_count = 0
            while RUNNING:
                readable, _, _ = select([sock], [], [], 1.0) # 1.0 second timeout
                if readable:
                    data, addr = sock.recvfrom(BUFFER_SIZE)
                    frame_count += 1
                    print(f"[CLIENT:VideoReceiver] Received video frame {frame_count} from {addr} ({len(data)} bytes)")
                    # In a real application, you would process or display this frame (e.g., Image.open(io.BytesIO(data)))
    except Exception as e:
        print(f"[CLIENT:VideoReceiver] Thread error: {e}")
    finally:
        print("[CLIENT:VideoReceiver] Thread stopped.")

def send_client_float_stream():
    """
    Thread function to send float arrays (NumPy arrays) to the server.
    """
    global RUNNING
    try:
        # Create a UDP socket for sending floats
        # Bind to client's IP, let OS pick ephemeral port for sending
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
            sock.bind((FEDORA_IP, 0)) # Bind to the specific interface IP
            print(f"[CLIENT:ClientFloatSender] Bound to {sock.getsockname()}")

            array_count = 0
            while RUNNING:
                float_array = generate_float_array()
                try:
                    # Pickle the NumPy array to bytes for transmission
                    serialized_array = pickle.dumps(float_array)
                    # Send the serialized array to the server's float receiving port
                    sock.sendto(serialized_array, (PI_IP, CLIENT_FLOAT_SEND_PORT))
                    array_count += 1
                    # print(f"[CLIENT:ClientFloatSender] Sent float array {array_count} ({len(serialized_array)} bytes)")
                except Exception as e:
                    print(f"[CLIENT:ClientFloatSender] Error sending float array: {e}")
                time.sleep(0.7) # Send array every 700ms
    except Exception as e:
        print(f"[CLIENT:ClientFloatSender] Thread error: {e}")
    finally:
        print("[CLIENT:ClientFloatSender] Thread stopped.")

def receive_server_float_stream():
    """
    Thread function to receive float arrays from the server.
    Uses select for non-blocking receive.
    """
    global RUNNING
    try:
        # Create a UDP socket for receiving floats from the server
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
            sock.bind((FEDORA_IP, SERVER_FLOAT_RECV_PORT)) # Bind to client's IP and specific port
            print(f"[CLIENT:ServerFloatReceiver] Listening on {FEDORA_IP}:{SERVER_FLOAT_RECV_PORT}")

            while RUNNING:
                readable, _, _ = select([sock], [], [], 1.0) # 1.0 second timeout
                if readable:
                    data, addr = sock.recvfrom(BUFFER_SIZE)
                    try:
                        # Unpickle the received bytes back into a NumPy array
                        received_array = pickle.loads(data)
                        print(f"[CLIENT:ServerFloatReceiver] Received float array from {addr}:")
                        # print(received_array) # Uncomment to see full array
                        print(f"  Shape: {received_array.shape}, Dtype: {received_array.dtype}")
                    except (pickle.UnpicklingError, ValueError) as e:
                        print(f"[CLIENT:ServerFloatReceiver] Error unpickling data from {addr}: {e}")
                    except Exception as e:
                        print(f"[CLIENT:ServerFloatReceiver] Unexpected error processing data from {addr}: {e}")
    except Exception as e:
        print(f"[CLIENT:ServerFloatReceiver] Thread error: {e}")
    finally:
        print("[CLIENT:ServerFloatReceiver] Thread stopped.")

def signal_handler(sig, frame):
    """Handles graceful shutdown on Ctrl+C."""
    global RUNNING
    print("\n[CLIENT] Ctrl+C detected. Shutting down gracefully...")
    RUNNING = False
    # Give threads a moment to finish their loops
    time.sleep(1) 
    sys.exit(0)

def main():
    signal.signal(signal.SIGINT, signal_handler)

    print(f"[{socket.gethostname()}] Starting Multi-Stream UDP Client...")

    # Start all communication threads
    video_receiver_thread = threading.Thread(target=receive_video_stream)
    client_float_sender_thread = threading.Thread(target=send_client_float_stream)
    server_float_receiver_thread = threading.Thread(target=receive_server_float_stream)

    video_receiver_thread.start()
    client_float_sender_thread.start()
    server_float_receiver_thread.start()

    # Keep the main thread alive until all other threads finish
    try:
        while RUNNING:
            time.sleep(0.5)
    except KeyboardInterrupt:
        pass
    finally:
        print("[CLIENT] Main thread exiting. Waiting for worker threads...")
        video_receiver_thread.join()
        client_float_sender_thread.join()
        server_float_receiver_thread.join()
        print("[CLIENT] All threads stopped. Client shut down.")

if __name__ == "__main__":
    main()

