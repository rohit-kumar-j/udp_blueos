from auv.auv import AUV

if __name__ == "__main__":
    auv_client = AUV(server=False, debug=True, camera_resolution=(640, 480), window=True, show_status_text=True)
    auv_client.run_main_loop()

    # auv_client = AUV(server=False, debug=True, camera_resolution=(640, 480), window=True, show_status_text=True)
    # control_thread = threading.Thread(target=client_interaction_example, args=(auv_client_with_control,), daemon=True)
    # control_thread.start()
    # auv_client_with_control.run_main_loop()
