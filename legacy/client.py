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
import io # Import io for BytesIO

# --- Video Display Specific Import ---
import cv2

# --- Configuration ---
PI_IP = '192.168.2.10'  # Raspberry Pi's static eth0 IP
FEDORA_IP = '192.168.2.1' # Fedora machine's static Ethernet IP

VIDEO_RECV_PORT = 65430        # Client receives video from Server on this port
CLIENT_FLOAT_SEND_PORT = 65431 # Client sends floats to Server on this port
SERVER_FLOAT_RECV_PORT = 65432 # Client receives floats from Server on this port

BUFFER_SIZE = 65536 # Max UDP packet size. Important for UDP.
                    # Adjust if your encoded frames exceed this. Max for IPv4 is ~65507 bytes.

# --- Global Flags for Thread Control ---
RUNNING = True
VIDEO_WINDOW_NAME = "Raspberry Pi Camera Stream"
SHOW_STATUS_TEXT = True # Global flag to control visibility of connection status text

# --- Colors for connection status display (BGR format for OpenCV) ---
COLOR_GREEN = (0, 255, 0) # Green for Connected
COLOR_RED = (0, 0, 255)   # Red for Disconnected

# --- Threshold for considering server disconnected (seconds) ---
DISCONNECT_THRESHOLD = 2.0 

def generate_float_array():
    """Generates a dummy NumPy array of floats."""
    return np.random.rand(5, 5).astype(np.float32) # Smaller array for client to server

def receive_video_stream():
    """
    Thread function to receive video frames from the server and display them.
    Uses select for non-blocking receive.
    """
    global RUNNING, SHOW_STATUS_TEXT
    last_frame_time = time.time() # Initialize last frame time
    is_connected = False
    
    # Initialize a placeholder for frame_decoded to use outside the if readable block
    frame_decoded = None

    try:
        # Create a UDP socket for receiving video
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
            sock.bind((FEDORA_IP, VIDEO_RECV_PORT)) # Bind to client's IP and specific port
            print(f"[CLIENT:VideoReceiver] Listening on {FEDORA_IP}:{VIDEO_RECV_PORT}")

            cv2.namedWindow(VIDEO_WINDOW_NAME, cv2.WINDOW_NORMAL)
            # Optional: cv2.resizeWindow(VIDEO_WINDOW_NAME, 640, 480) # Adjust as needed

            frame_count = 0
            while RUNNING:
                # Check if window was closed by user
                if cv2.getWindowProperty(VIDEO_WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
                    print(f"[CLIENT:VideoReceiver] Video window '{VIDEO_WINDOW_NAME}' closed. Terminating stream.")
                    RUNNING = False
                    break # Exit the loop if window is closed

                readable, _, _ = select([sock], [], [], 0.01) # Small timeout for responsiveness
                
                # Check for new data
                if readable:
                    data, addr = sock.recvfrom(BUFFER_SIZE)
                    last_frame_time = time.time() # Update last frame time
                    is_connected = True
                    frame_count += 1
                    # print(f"[CLIENT:VideoReceiver] Received video frame {frame_count} from {addr} ({len(data)} bytes)")

                    try:
                        # Decode JPEG bytes to a NumPy array using OpenCV
                        np_array = np.frombuffer(data, np.uint8)
                        frame_decoded = cv2.imdecode(np_array, cv2.IMREAD_COLOR) # IMREAD_COLOR for RGB/BGR

                        if frame_decoded is None:
                            print(f"[CLIENT:VideoReceiver] Failed to decode frame {frame_count} from {addr}.")
                            # If decode fails, set frame_decoded to a blank frame to avoid errors later
                            frame_decoded = np.zeros((480, 640, 3), dtype=np.uint8) 

                    except Exception as e:
                        print(f"[CLIENT:VideoReceiver] Error processing video frame: {e}")
                        frame_decoded = np.zeros((480, 640, 3), dtype=np.uint8) # Fallback to blank frame on error
                
                # Determine connection status and text to display
                if time.time() - last_frame_time > DISCONNECT_THRESHOLD:
                    if is_connected: # Only print message once when status changes
                        print("[CLIENT:VideoReceiver] Server not sending data (Disconnected).")
                    is_connected = False
                    status_text = "Disconnected"
                    text_color = COLOR_RED
                    # If disconnected and no frame decoded recently, use a black frame
                    if frame_decoded is None or not is_connected:
                         frame_to_display = np.zeros((480, 640, 3), dtype=np.uint8)
                    else: # If a frame was decoded, but now disconnected, use that frame to draw on
                         frame_to_display = frame_decoded.copy() # Use a copy to avoid modifying original
                else:
                    is_connected = True
                    status_text = "Connected"
                    text_color = COLOR_GREEN
                    if frame_decoded is None: # If connected but no frame yet, use a black frame
                        frame_to_display = np.zeros((480, 640, 3), dtype=np.uint8)
                    else:
                        frame_to_display = frame_decoded # Use the most recently decoded frame

                # Draw connection status text if the flag is True
                if SHOW_STATUS_TEXT:
                    # Calculate text position to be somewhat centered/visible near top
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    font_scale = 1
                    font_thickness = 2
                    text_size = cv2.getTextSize(status_text, font, font_scale, font_thickness)[0]
                    
                    # Position slightly from top-left, adjust for text height
                    text_x = 10
                    text_y = text_size[1] + 10 # 10 pixels from top, then text height

                    cv2.putText(frame_to_display, status_text, (text_x, text_y), 
                                font, font_scale, text_color, font_thickness, cv2.LINE_AA)
                
                # Display the frame with status text
                cv2.imshow(VIDEO_WINDOW_NAME, frame_to_display)

                # Always process window events to detect 'q' or window close
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    print("[CLIENT:VideoReceiver] 'q' pressed, stopping video stream.")
                    RUNNING = False

    except Exception as e:
        print(f"[CLIENT:VideoReceiver] Thread error: {e}")
    finally:
        print("[CLIENT:VideoReceiver] Destroying video window...")
        cv2.destroyAllWindows() # Close all OpenCV windows
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
    # Give threads a moment to finish their loops and close windows
    time.sleep(1)
    # Ensure all OpenCV windows are destroyed on exit
    cv2.destroyAllWindows() 
    sys.exit(0)

def main():
    signal.signal(signal.SIGINT, signal_handler)

    print(f"[{socket.gethostname()}] Starting Multi-Stream UDP Client (with Video Display!)...")

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
            # You can toggle SHOW_STATUS_TEXT dynamically from here if needed for debugging
            # For example, after 10 seconds:
            # if time.time() - start_time > 10 and SHOW_STATUS_TEXT:
            #     global SHOW_STATUS_TEXT
            #     SHOW_STATUS_TEXT = False
            #     print("[CLIENT] Toggling status text OFF.")
            time.sleep(0.5)
    except KeyboardInterrupt:
        # This block might not be hit if signal_handler catches first
        pass
    finally:
        print("[CLIENT] Main thread exiting. Waiting for worker threads...")
        # Join threads with a timeout, so they don't hang indefinitely if something goes wrong
        video_receiver_thread.join(timeout=5)
        client_float_sender_thread.join(timeout=5)
        server_float_receiver_thread.join(timeout=5)
        print("[CLIENT] All threads stopped. Client shut down.")

if __name__ == "__main__":
    main()

