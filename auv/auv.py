# auv.py
# This module can be used as a server on the Raspberry Pi or a client on the Fedora machine.

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

# --- Video/Camera Specific Imports ---
import cv2
try:
    # This import is only needed on the Raspberry Pi for camera capture
    from picamera2 import Picamera2
    from libcamera import controls
    PICAMERA2_AVAILABLE = True
except ImportError:
    PICAMERA2_AVAILABLE = False

# --- Hardware-specific Imports (moved to top as requested) ---
# These will only successfully import on the Raspberry Pi with the drivers installed.
# They are wrapped in a try-except block at initialization of the AUV class.
try:
    from icm20602 import ICM20602
    ICM20602_AVAILABLE = True
except ImportError:
    ICM20602_AVAILABLE = False
except Exception as e:
    # Catch any other error during import (e.g., missing dependencies of icm20602)
    print(f"Warning: Could not import ICM20602. Error: {e}")
    ICM20602_AVAILABLE = False

try:
    from pca9685 import PCA9685
    PCA9685_AVAILABLE = True
except ImportError:
    PCA9685_AVAILABLE = False
except Exception as e:
    # Catch any other error during import (e.g., missing dependencies of pca9685)
    print(f"Warning: Could not import PCA9685. Error: {e}")
    PCA9685_AVAILABLE = False


# --- Configuration Constants (Shared) ---
PI_IP = '192.168.2.10'      # Raspberry Pi's static eth0 IP
FEDORA_IP = '192.168.2.1'   # Fedora machine's static Ethernet IP

# Ports for UDP Communication
VIDEO_STREAM_PORT = 65430        # Server (Pi) sends video to Client (Fedora) / Client receives video from Server
CLIENT_CMD_SEND_PORT = 65431     # Client (Fedora) sends motor commands (floats) to Server (Pi)
SERVER_FLOAT_SEND_PORT = 65432   # Server (Pi) sends general float data to Client (Fedora)
SERVER_IMU_SEND_PORT = 65433     # Server (Pi) sends IMU data (floats) to Client (Fedora)

BUFFER_SIZE = 65507 # Max UDP payload size (IPv4). Slightly less than 65536 to account for headers.

# --- Global Settings for Client-side Display ---
VIDEO_WINDOW_NAME = "Raspberry Pi Camera Stream"
COLOR_GREEN = (0, 255, 0) # Green for Connected status text (BGR)
COLOR_RED = (0, 0, 255)   # Red for Disconnected status text (BGR)
DISCONNECT_THRESHOLD = 2.0 # Seconds without a video frame to consider disconnected


class AUV:
    def __init__(self, server=True, debug=False, camera_resolution=(640, 480), window=True, show_status_text=True):
        """
        Initializes the AUV communication module.

        Args:
            server (bool): True to initialize as a server (Raspberry Pi), False as a client (Fedora).
            debug (bool): If True, enables print statements for debugging.
            camera_resolution (tuple): Desired resolution for camera capture (width, height).
            window (bool): If True (default for client), displays the video feed in an OpenCV window.
                           If False, no video window is rendered on the client.
            show_status_text (bool): If True (default), displays "Connected"/"Disconnected" text on the video feed.
                                     Only applicable if `window` is also True.
        """
        self.server = server
        self.debug = debug
        self.running = True  # Flag to control thread execution
        self.running_event = threading.Event() # Event for cleaner thread shutdown
        self.running_event.set() # Set the event initially to signal threads to run

        self.camera_resolution = camera_resolution
        self.show_video_window = window # New flag for client-side window rendering
        self.show_status_text = show_status_text # New flag for status text visibility

        self.threads = []
        self.sockets = {}

        # Hardware interfaces (initialized conditionally)
        self.imu_sensor = None
        self.motor_controller = None

        # --- Client-side data storage ---
        self.latest_camera_frame = None  # Stores the latest decoded camera frame (NumPy array)
        self.latest_imu_data = None      # Stores the latest received IMU data (NumPy array)
        self.latest_server_float_data = None # Stores the latest received general float data from server

        # --- Server-side data storage ---
        self.latest_motor_command = None # Stores the latest received motor command (NumPy array)

        # Determine local and remote IPs based on role
        self.local_ip = PI_IP if self.server else FEDORA_IP
        self.remote_ip = FEDORA_IP if self.server else PI_IP

        self._log(f"AUV instance created as {'SERVER' if self.server else 'CLIENT'} on {self.local_ip}. Debugging: {self.debug}")
        if not self.server:
            self._log(f"Video window rendering: {'ENABLED' if self.show_video_window else 'DISABLED'}")
            if self.show_video_window:
                self._log(f"Connection status text: {'ENABLED' if self.show_status_text else 'DISABLED'}")

        # Initialize hardware components if this is the server
        if self.server:
            self._init_hardware()

        self._init_sockets()
        self._start_communication_threads()
        self._register_signal_handler()

    def _log(self, message):
        """Prints a message if debugging is enabled."""
        if self.debug:
            print(f"[AUV:{'SERVER' if self.server else 'CLIENT'}] {message}")

    def _init_hardware(self):
        """Initializes hardware components (IMU, Motor Controller) on the server."""
        self._log("Attempting to initialize hardware components...")
        if ICM20602_AVAILABLE:
            try:
                self.imu_sensor = ICM20602()
                self._log("ICM20602 IMU sensor initialized successfully.")
            except Exception as e:
                self._log(f"Error initializing ICM20602: {e}. IMU data will be dummy.")
                self.imu_sensor = None
        else:
            self._log("ICM20602 module not available. IMU data will be dummy.")

        if PCA9685_AVAILABLE:
            try:
                self.motor_controller = PCA9685(bus=1)
                self.motor_controller.set_pwm_frequency(50) # Set a common PWM frequency for motors (e.g., 50 Hz)
                self.motor_controller.output_enable()
                self._log("PCA9685 motor controller initialized successfully.")
            except Exception as e:
                self._log(f"Error initializing PCA9685: {e}. Motor commands will not be applied.")
                self.motor_controller = None
        else:
            self._log("PCA9685 module not available. Motor commands will not be applied.")


    def _init_sockets(self):
        """Initializes all necessary UDP sockets for communication."""
        try:
            # Client (Fedora) to Server (Pi) - Motor Commands (Sending)
            if not self.server: # Client
                # Client sends from an ephemeral port, so bind to 0
                self.sockets['client_cmd_send'] = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
                self.sockets['client_cmd_send'].bind((self.local_ip, 0))
                self._log(f"Client Cmd Sender bound to {self.sockets['client_cmd_send'].getsockname()}")
            else: # Server
                # Server receives motor commands on a specific port
                self.sockets['client_cmd_recv'] = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
                self.sockets['client_cmd_recv'].bind((self.local_ip, CLIENT_CMD_SEND_PORT))
                self._log(f"Client Cmd Receiver listening on {self.local_ip}:{CLIENT_CMD_SEND_PORT}")

            # Server (Pi) to Client (Fedora) - Video Stream (Sending/Receiving)
            if self.server: # Server
                self.sockets['video_send'] = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
                self.sockets['video_send'].bind((self.local_ip, 0)) # Ephemeral port for sending
                self._log(f"Video Sender bound to {self.sockets['video_send'].getsockname()}")
            else: # Client
                self.sockets['video_recv'] = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
                # Client now binds to VIDEO_STREAM_PORT
                self.sockets['video_recv'].bind((self.local_ip, VIDEO_STREAM_PORT)) 
                self._log(f"Video Receiver listening on {self.local_ip}:{VIDEO_STREAM_PORT}") # Updated log

            # Server (Pi) to Client (Fedora) - General Float Stream (Sending/Receiving)
            if self.server: # Server
                self.sockets['server_float_send'] = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
                self.sockets['server_float_send'].bind((self.local_ip, 0)) # Ephemeral port for sending
                self._log(f"Server Float Sender bound to {self.sockets['server_float_send'].getsockname()}")
            else: # Client
                self.sockets['server_float_recv'] = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
                self.sockets['server_float_recv'].bind((self.local_ip, SERVER_FLOAT_SEND_PORT))
                self._log(f"Server Float Receiver listening on {self.local_ip}:{SERVER_FLOAT_SEND_PORT}")

            # Server (Pi) to Client (Fedora) - IMU Data Stream (Sending/Receiving) - NEW
            if self.server: # Server
                self.sockets['imu_send'] = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
                self.sockets['imu_send'].bind((self.local_ip, 0)) # Ephemeral port for sending
                self._log(f"IMU Data Sender bound to {self.sockets['imu_send'].getsockname()}")
            else: # Client
                self.sockets['imu_recv'] = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
                self.sockets['imu_recv'].bind((self.local_ip, SERVER_IMU_SEND_PORT))
                self._log(f"IMU Data Receiver listening on {self.local_ip}:{SERVER_IMU_SEND_PORT}")

        except OSError as e:
            self._log(f"CRITICAL ERROR: Could not bind sockets. Check IP addresses, port availability, and firewalls. Error: {e}")
            self.stop() # Stop further initialization if sockets fail
            sys.exit(1)


    def _start_communication_threads(self):
        """Starts threads based on whether the instance is a server or client."""
        if self.server:
            # Server communication threads
            self.threads.append(threading.Thread(target=self._send_camera_stream, daemon=True))
            self.threads.append(threading.Thread(target=self._send_imu_data, daemon=True)) # New
            self.threads.append(threading.Thread(target=self._send_server_float_stream, daemon=True))
            self.threads.append(threading.Thread(target=self._receive_motor_commands, daemon=True))
        else:
            # Client communication threads
            self.threads.append(threading.Thread(target=self._receive_camera_stream, daemon=True))
            self.threads.append(threading.Thread(target=self._receive_imu_data, daemon=True))
            self.threads.append(threading.Thread(target=self._receive_server_float_stream, daemon=True))
            # Client motor command sending is now on-demand via send_motor_cmd()

        for t in self.threads:
            t.start()
            self._log(f"Started thread: {t.name}")

    def _register_signal_handler(self):
        """Registers the signal handler for graceful shutdown."""
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler) # Also handle termination signal

    def _signal_handler(self, sig, frame):
        """Handles graceful shutdown on Ctrl+C or termination signals."""
        self._log(f"Signal {sig} detected. Shutting down gracefully...")
        self.stop()
        sys.exit(0)

    def stop(self):
        """Sets the running flag to False and attempts to join all threads."""
        self.running = False # Signal threads to stop their main loops
        self.running_event.clear() # Signal threads blocking on select() to exit

        self._log("Stopping all communication threads...")
        
        # Wait for threads to finish their current operations and exit gracefully
        for t in self.threads:
            if t.is_alive():
                t.join(timeout=2) # Give threads a moment to finish
                if t.is_alive():
                    self._log(f"Warning: Thread {t.name} did not terminate gracefully.")
        
        # Now that threads have (hopefully) exited their blocking calls, close sockets
        for sock in self.sockets.values():
            try:
                # Shutdown both read and write ends to unblock blocking calls
                sock.shutdown(socket.SHUT_RDWR)
            except OSError as e:
                # Ignore "Transport endpoint is not connected" or "Socket is not connected"
                # which can happen if socket wasn't fully connected or already closed
                # Also ignore EBADF (9) if the socket was already implicitly closed
                if e.errno not in (107, 57, 9): # 107=ENOTCONN, 57=ENOTCONN (macOS), 9=EBADF (Bad File Descriptor)
                    self._log(f"Warning: Error during socket shutdown for {sock.getsockname()}: {e}")
            finally:
                sock.close() # Ensure the socket is ultimately closed

        self._log("All threads signaled to stop and sockets closed.")
        if not self.server and self.show_video_window: # Only destroy if client and window was active
            cv2.destroyAllWindows() # Ensure any OpenCV windows are closed

    # --- Server-side Communication Threads ---

    def _send_camera_stream(self):
        """Server: Captures video frames using OpenCV or picamera2 and sends to client."""
        cap = None
        picam2 = None
        # Moved try-except for camera initialization outside the loop
        try:
            if PICAMERA2_AVAILABLE:
                self._log("Initializing camera with picamera2...")
                picam2 = Picamera2()
                camera_config = picam2.create_video_configuration(main={"size": self.camera_resolution, "format": "RGB888"})
                picam2.configure(camera_config)
                picam2.start()
                self._log("Picamera2 started.")
            else:
                self._log("Initializing camera with OpenCV VideoCapture...")
                cap = cv2.VideoCapture(0) # Use 0 for default camera
                if not cap.isOpened():
                    raise IOError("Cannot open camera with OpenCV VideoCapture (or no camera found).")
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.camera_resolution[0])
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.camera_resolution[1])
                cap.set(cv2.CAP_PROP_FPS, 30) # Attempt to set 30 FPS
                self._log("OpenCV VideoCapture started.")

            sock = self.sockets['video_send']
            self._log(f"Server: Video Sender sending to {self.remote_ip}:{VIDEO_STREAM_PORT}")

            frame_count = 0
            while self.running_event.is_set(): # Use event to control loop
                frame = None
                if PICAMERA2_AVAILABLE:
                    frame = picam2.capture_array()
                else:
                    ret, frame = cap.read()
                    if not ret:
                        self._log("Server: Failed to grab frame from OpenCV, retrying...")
                        time.sleep(0.01)
                        continue

                encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 20] # REDUCED QUALITY TO AVOID "MESSAGE TOO LONG"
                _, frame_bytes_np = cv2.imencode('.jpg', frame, encode_param) # imencode returns a numpy array
                frame_bytes = frame_bytes_np.tobytes() # Get the actual bytes from the numpy array

                if len(frame_bytes) > BUFFER_SIZE:
                    self._log(f"Server: Warning! Frame size ({len(frame_bytes)} bytes) exceeds BUFFER_SIZE ({BUFFER_SIZE} bytes). This frame will be truncated or dropped. Consider lowering resolution or JPEG quality further.")

                try:
                    sock.sendto(frame_bytes, (self.remote_ip, VIDEO_STREAM_PORT))
                    frame_count += 1
                    # self._log(f"Server: Sent video frame {frame_count} ({len(frame_bytes)} bytes)")
                except Exception as e:
                    # Only log errors if not in shutdown process (socket already closed)
                    if self.running_event.is_set():
                        self._log(f"Server: Error sending video frame: {e}")
                
                time.sleep(0.03) # ~33 FPS. Adjust based on network/CPU.

        except Exception as e: # Catch camera initialization errors
            self._log(f"Server: Video Stream Thread critical error during camera init/loop: {e}")
            self.running_event.clear() # Force shutdown if camera fails critically
        finally:
            if picam2 and PICAMERA2_AVAILABLE:
                self._log("Server: Stopping picamera2...")
                picam2.stop()
                picam2.close()
            if cap and cap.isOpened():
                self._log("Server: Releasing OpenCV camera...")
                cap.release()
            self._log("Server: Video Sender thread stopped.")

    def _send_imu_data(self):
        """Server: Sends IMU data (NumPy array) to the client."""
        sock = self.sockets['imu_send']
        self._log(f"Server: IMU Data Sender sending to {self.remote_ip}:{SERVER_IMU_SEND_PORT}")
        
        imu_count = 0
        while self.running_event.is_set(): # Use event to control loop
            imu_data_array = None
            if self.imu_sensor: # Check if sensor was initialized successfully
                try:
                    data = self.imu_sensor.read_all()
                    # Example: Concatenate accelerometer and gyroscope data
                    imu_data_array = np.array([
                        data.a.x, data.a.y, data.a.z,
                        data.g.x, data.g.y, data.g.z,
                        data.t # Include temperature
                    ], dtype=np.float32)
                except Exception as e: # Catch errors during sensor reading
                    self._log(f"Server: Error reading from ICM20602: {e}. Sending dummy IMU data.")
                    imu_data_array = np.random.rand(7).astype(np.float32) * 10 - 5 # Dummy data (7 values: 6 DOF + temp)
            
            # If sensor not available or failed reading, use dummy data
            if imu_data_array is None:
                imu_data_array = np.random.rand(7).astype(np.float32) * 10 - 5

            try:
                serialized_imu = pickle.dumps(imu_data_array)
                sock.sendto(serialized_imu, (self.remote_ip, SERVER_IMU_SEND_PORT))
                imu_count += 1
                # self._log(f"Server: Sent IMU data {imu_count} ({len(serialized_imu)} bytes)")
            except Exception as e: # Catch errors during socket send
                if self.running_event.is_set():
                    self._log(f"Server: Error sending IMU data: {e}")
            time.sleep(0.05) # Send IMU data 20 times per second

        self._log("Server: IMU Data Sender thread stopped.")

    def _send_server_float_stream(self):
        """Server: Sends general float arrays (NumPy array) to the client."""
        sock = self.sockets['server_float_send']
        self._log(f"Server: Server Float Sender sending to {self.remote_ip}:{SERVER_FLOAT_SEND_PORT}")
        
        array_count = 0
        while self.running_event.is_set(): # Use event to control loop
            float_array = np.random.rand(10, 10).astype(np.float32) # 10x10 array
            try:
                serialized_array = pickle.dumps(float_array)
                sock.sendto(serialized_array, (self.remote_ip, SERVER_FLOAT_SEND_PORT))
                array_count += 1
                # self._log(f"Server: Sent general float array {array_count} ({len(serialized_array)} bytes)")
            except Exception as e:
                if self.running_event.is_set():
                    self._log(f"Server: Error sending general float array: {e}")
            time.sleep(0.5) # Send array every 500ms

        self._log("Server: Server Float Sender thread stopped.")

    def _receive_motor_commands(self):
        """Server: Receives motor command (float array) from the client and applies them."""
        sock = self.sockets['client_cmd_recv']
        self._log(f"Server: Motor Command Receiver listening on {self.local_ip}:{CLIENT_CMD_SEND_PORT}")
        
        while self.running_event.is_set(): # Use event to control loop
            # Use select with a timeout to allow the loop to check self.running_event.is_set()
            # without blocking indefinitely on recvfrom.
            readable, _, _ = select([sock], [], [], 0.01) # Small timeout for responsiveness
            if readable:
                try: # Only try to recvfrom if readable, otherwise it would block
                    data, addr = sock.recvfrom(BUFFER_SIZE)
                    received_array = pickle.loads(data)
                    self.latest_motor_command = received_array # Store the latest command
                    self._log(f"Server: Received motor command from {addr}: {received_array.shape}, Dtype: {received_array.dtype}")
                    
                    if self.motor_controller: # Check if controller was initialized successfully
                        if received_array.ndim == 1 and received_array.shape[0] <= 16: # Max 16 channels for PCA9685
                            for i, val in enumerate(received_array):
                                # Ensure value is clipped to -1.0 to 1.0 range
                                clipped_val = np.clip(val, -1.0, 1.0) 
                                pwm_us = 1500 + clipped_val * 500 # Maps -1 to 1000, 0 to 1500, 1 to 2000
                                self.motor_controller.channel_set_pwm(i, pwm_us)
                                
                            self._log(f"Server: Applied motor commands: {received_array}")
                        else:
                            self._log(f"Server: Received motor command shape {received_array.shape} not supported for PCA9685 application.")
                    else:
                        self._log("Server: Motor controller not initialized, commands not applied.")

                except (pickle.UnpicklingError, ValueError) as e:
                    self._log(f"Server: Error unpickling motor command from {addr}: {e}")
                except OSError as e: # Specifically catch OSError (e.g., Bad file descriptor)
                    if e.errno == 9: # Bad file descriptor, likely socket closed during shutdown
                        self._log(f"Server: Socket error during motor command reception (likely shutdown): {e}")
                        break # Exit loop cleanly
                    else:
                        self._log(f"Server: Unexpected OSError during motor command processing from {addr}: {e}")
                except Exception as e: # Catch other unexpected errors
                    self._log(f"Server: Unexpected error processing motor command from {addr}: {e}")
            else:
                # If select timed out and running_event is cleared, break the loop
                if not self.running_event.is_set():
                    break
        
        if self.motor_controller:
            self._log("Server: Disabling motor outputs...")
            self.motor_controller.output_disable() # Disable motors on shutdown
        self._log("Server: Motor Command Receiver thread stopped.")

    # --- Public API for Client ---

    def _receive_camera_stream(self):
        """Client: Receives video frames from the server and (optionally) displays them."""
        sock = self.sockets['video_recv']
        self._log(f"Client: Video Receiver listening on {self.local_ip}:{VIDEO_STREAM_PORT}")
        
        last_frame_time = time.time()
        is_connected = False
        # Initialize frame_to_display with the correct resolution/channels
        frame_to_display = np.zeros((self.camera_resolution[1], self.camera_resolution[0], 3), dtype=np.uint8) 
        
        if self.show_video_window:
            cv2.namedWindow(VIDEO_WINDOW_NAME, cv2.WINDOW_NORMAL)
            # cv2.resizeWindow(VIDEO_WINDOW_NAME, self.camera_resolution[0], self.camera_resolution[1]) # Set initial window size

        while self.running_event.is_set(): # Use event to control loop
            if self.show_video_window:
                # Check if window was closed by user only if window is being displayed
                if cv2.getWindowProperty(VIDEO_WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
                    self._log(f"Client: Video window '{VIDEO_WINDOW_NAME}' closed. Terminating client.")
                    self.running_event.clear() # Signal all threads to stop
                    break # Exit the loop if window is closed

            # Use select with a timeout to allow the loop to check self.running_event.is_set()
            # without blocking indefinitely on recvfrom.
            readable, _, _ = select([sock], [], [], 0.01) # Small timeout for responsiveness
            
            if readable:
                try: # Only try to recvfrom if readable, otherwise it would block
                    data, addr = sock.recvfrom(BUFFER_SIZE)
                    last_frame_time = time.time()
                    is_connected = True
                    
                    np_array = np.frombuffer(data, np.uint8)
                    decoded_frame = cv2.imdecode(np_array, cv2.IMREAD_COLOR)

                    if decoded_frame is not None:
                        # Update the latest camera frame for camera() method access
                        self.latest_camera_frame = decoded_frame
                        frame_to_display = decoded_frame # Use this for drawing status if window is shown
                    else:
                        self._log(f"Client: Failed to decode frame. Data length: {len(data)} from {addr}.")
                        # Fallback to black frame if decode fails, for display
                        if self.show_video_window:
                            frame_to_display = np.zeros((self.camera_resolution[1], self.camera_resolution[0], 3), dtype=np.uint8) 

                except OSError as e: # Specifically catch OSError (e.g., Bad file descriptor)
                    if e.errno == 9: # Bad file descriptor, likely socket closed during shutdown
                        self._log(f"Client: Socket error during camera reception (likely shutdown): {e}")
                        break # Exit loop cleanly
                    else:
                        self._log(f"Client: Unexpected OSError during video frame processing from {addr}: {e}")
                except Exception as e: # Catch other unexpected errors
                    self._log(f"Client: Error processing video frame: {e}")
                    # Fallback to black frame on error
                    if self.show_video_window:
                        frame_to_display = np.zeros((self.camera_resolution[1], self.camera_resolution[0], 3), dtype=np.uint8) 
            else:
                # If select timed out and running_event is cleared, break the loop
                if not self.running_event.is_set():
                    break
            
            # Determine connection status and text to display (only if window is shown)
            if self.show_video_window:
                if time.time() - last_frame_time > DISCONNECT_THRESHOLD:
                    if is_connected:
                        self._log("Client: Server not sending video data (Disconnected).")
                    is_connected = False
                    status_text = "Disconnected"
                    text_color = COLOR_RED
                    # If disconnected and no valid frame, ensure a black frame is used
                    if frame_to_display.shape[0] == 0 or frame_to_display.shape[1] == 0: # Check if frame is empty
                        frame_to_display = np.zeros((self.camera_resolution[1], self.camera_resolution[0], 3), dtype=np.uint8)
                else:
                    is_connected = True
                    status_text = "Connected"
                    text_color = COLOR_GREEN
                    # If connected but no frame yet, ensure a black frame is used
                    if frame_to_display.shape[0] == 0 or frame_to_display.shape[1] == 0:
                        frame_to_display = np.zeros((self.camera_resolution[1], self.camera_resolution[0], 3), dtype=np.uint8)

                # Draw connection status text if enabled
                if self.show_status_text: # Changed to self.show_status_text
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    font_scale = 1
                    font_thickness = 2
                    text_size = cv2.getTextSize(status_text, font, font_scale, font_thickness)[0]
                    text_x = 10
                    text_y = text_size[1] + 10

                    cv2.putText(frame_to_display, status_text, (text_x, text_y),
                                font, font_scale, text_color, font_thickness, cv2.LINE_AA)
                
                cv2.imshow(VIDEO_WINDOW_NAME, frame_to_display)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    self._log("Client: 'q' pressed, stopping video stream.")
                    self.running_event.clear() # Signal all threads to stop

        self._log("Client: Video Receiver thread stopped.")

    def _receive_imu_data(self):
        """Client: Receives IMU data (NumPy array) from the server."""
        sock = self.sockets['imu_recv']
        self._log(f"Client: IMU Data Receiver listening on {self.local_ip}:{SERVER_IMU_SEND_PORT}")
        
        while self.running_event.is_set(): # Use event to control loop
            readable, _, _ = select([sock], [], [], 0.01) # Small timeout for responsiveness
            if readable:
                try:
                    data, addr = sock.recvfrom(BUFFER_SIZE)
                    received_array = pickle.loads(data)
                    self.latest_imu_data = received_array # Store the latest IMU data
                    self._log(f"Client: Received IMU data from {addr}: Data:{received_array}, {received_array.shape}, Dtype: {received_array.dtype}")
                except (pickle.UnpicklingError, ValueError) as e:
                    self._log(f"Client: Error unpickling IMU data from {addr}: {e}")
                except OSError as e: # Specifically catch OSError (e.g., Bad file descriptor)
                    if e.errno == 9: # Bad file descriptor, likely socket closed during shutdown
                        self._log(f"Client: Socket error during IMU reception (likely shutdown): {e}")
                        break # Exit loop cleanly
                    else:
                        self._log(f"Client: Unexpected OSError during IMU data processing from {addr}: {e}")
                except Exception as e:
                    self._log(f"Client: Unexpected error processing IMU data from {addr}: {e}")
            else:
                # If select timed out and running_event is cleared, break the loop
                if not self.running_event.is_set():
                    break
        self._log("Client: IMU Data Receiver thread stopped.")

    def _receive_server_float_stream(self):
        """Client: Receives general float arrays from the server."""
        sock = self.sockets['server_float_recv']
        self._log(f"Client: Server Float Receiver listening on {self.local_ip}:{SERVER_FLOAT_SEND_PORT}")
        
        while self.running_event.is_set(): # Use event to control loop
            readable, _, _ = select([sock], [], [], 0.01) # Small timeout for responsiveness
            if readable:
                try:
                    data, addr = sock.recvfrom(BUFFER_SIZE)
                    received_array = pickle.loads(data)
                    self.latest_server_float_data = received_array # Store the latest general float data
                    self._log(f"Client: Received general float array from {addr}: {received_array.shape}, Dtype: {received_array.dtype}")
                except (pickle.UnpicklingError, ValueError) as e:
                    self._log(f"Client: Error unpickling general float data from {addr}: {e}")
                except OSError as e: # Specifically catch OSError (e.g., Bad file descriptor)
                    if e.errno == 9: # Bad file descriptor, likely socket closed during shutdown
                        self._log(f"Client: Socket error during float reception (likely shutdown): {e}")
                        break # Exit loop cleanly
                    else:
                        self._log(f"Client: Unexpected OSError during float data processing from {addr}: {e}")
                except Exception as e:
                    self._log(f"Client: Unexpected error processing general float data from {addr}: {e}")
            else:
                # If select timed out and running_event is cleared, break the loop
                if not self.running_event.is_set():
                    break
        self._log("Client: Server Float Receiver thread stopped.")

    # --- Public API for Client ---

    def imu(self):
        """
        Client method: Returns the latest IMU data received from the server.
        Raises ValueError if called on the server.
        The returned array will be a NumPy array of shape (7,) with float32 dtype:
        [accel_x, accel_y, accel_z, gyro_x, gyro_y, gyro_z, temperature]
        """
        if self.server:
            raise ValueError("imu() can only be called on a client instance.")
        
        if self.latest_imu_data is None:
            self._log("Client: No IMU data received yet.")
        return self.latest_imu_data # Return the stored IMU data

    def send_motor_cmd(self, motor_data: np.ndarray):
        """
        Client method: Sends motor command data (NumPy array) to the server.
        Raises ValueError if called on the server or if motor_data is not a NumPy array.
        The `motor_data` array should contain values suitable for your motor control,
        e.g., 4 float32 values representing thrust/direction for 4 motors.

        Args:
            motor_data (np.ndarray): A NumPy array containing motor command values.
                                     Expected shape: (N,) where N is number of motors.
                                     Expected dtype: np.float32.
        """
        if self.server:
            raise ValueError("send_motor_cmd() can only be called on a client instance.")
        if not isinstance(motor_data, np.ndarray):
            raise TypeError("motor_data must be a NumPy array.")
        
        sock = self.sockets['client_cmd_send']
        try:
            serialized_cmd = pickle.dumps(motor_data)
            sock.sendto(serialized_cmd, (self.remote_ip, CLIENT_CMD_SEND_PORT))
            self._log(f"Client: Sent motor command: {motor_data.shape}, Dtype: {motor_data.dtype}")
        except Exception as e:
            # Only log errors if not in shutdown process
            if self.running_event.is_set():
                self._log(f"Client: Error sending motor command: {e}")

    def camera(self):
        """
        Client method: Returns the latest camera frame received from the server.
        Raises ValueError if called on the server.
        The returned frame is a NumPy array (OpenCV format, BGR).
        """
        if self.server:
            raise ValueError("camera() can only be called on a client instance.")
        
        if self.latest_camera_frame is None:
            self._log("Client: No camera frame received yet.")
        return self.latest_camera_frame # Return the stored frame

    def run_main_loop(self):
        """
        Runs the main loop of the AUV instance, keeping it alive.
        This method should be called after initializing the AUV object.
        """
        self._log("Starting AUV main loop...")
        try:
            while self.running: # This loop keeps the main thread alive, controlled by self.running
                # The communication is handled by threads, main loop just keeps process alive
                time.sleep(0.01) # Small sleep to prevent busy-waiting
        except KeyboardInterrupt:
            self._log("KeyboardInterrupt detected in main loop.")
        finally:
            self.stop() # Ensure graceful shutdown on exit
            self._log("AUV main loop exited.")

# --- Example Usage (for testing the module) ---
if __name__ == "__main__":
    # --- IMPORTANT: Configure IP addresses correctly for your setup ---
    # To run on Raspberry Pi (Server):
    # This will initialize the camera and IMU drivers (if available)
    # and wait for motor commands from the client.
    # auv_server = AUV(server=True, debug=True, camera_resolution=(640, 480))
    # auv_server.run_main_loop()

    # To run on Fedora Machine (Client):
    # This will display the video feed, receive IMU data, and allow sending motor commands.
    auv_client = AUV(server=False, debug=True, camera_resolution=(640, 480), window=True, show_status_text=True)
    auv_client.run_main_loop()

    # Example of client interaction in a separate thread/process (if not using run_main_loop directly)
    # This demonstrates how you'd use the public methods
    # def client_interaction_example(auv_instance):
    #     motor_cmd_idx = 0
    #     while auv_instance.running:
    #         # Get latest camera frame (if window=False, you'd get it here to process)
    #         frame = auv_instance.camera()
    #         if frame is not None:
    #             # You can process this frame here (e.g., run object detection)
    #             pass
    #         
    #         # Get latest IMU data
    #         imu_data = auv_instance.imu()
    #         if imu_data is not None:
    #             auv_instance._log(f"Client: Latest IMU Data: {imu_data}")
    #
    #         # Send dummy motor commands (e.g., for 4 motors)
    #         # Values typically -1.0 to 1.0 (for thrust) or 1000-2000 for PWM
    #         if motor_cmd_idx % 2 == 0:
    #             motor_command = np.array([1.0, 0.5, 0.0, -0.5], dtype=np.float32) # Example forward/turn
    #         else:
    #             motor_command = np.array([-1.0, -0.5, 0.0, 0.5], dtype=np.float32) # Example reverse/turn
    #         auv_instance.send_motor_cmd(motor_command)
    #         motor_cmd_idx += 1
    #
    #         time.sleep(1.0) # Send command every second
    #
    # # To run this example interaction along with the video window:
    # # auv_client_with_control = AUV(server=False, debug=True, camera_resolution=(640, 480), window=True, show_status_text=True)
    # # control_thread = threading.Thread(target=client_interaction_example, args=(auv_client_with_control,), daemon=True)
    # # control_thread.start()
    # # auv_client_with_control.run_main_loop()

