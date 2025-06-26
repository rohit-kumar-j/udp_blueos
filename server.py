from auv.auv import AUV

if __name__ == "__main__":
    auv_server = AUV(server=True, debug=True, camera_resolution=(640, 480))
    auv_server.run_main_loop()


