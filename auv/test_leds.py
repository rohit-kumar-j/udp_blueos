# auv.py
from flashlights import setup_led, set_brightness
import time

channel = setup_led(7)
set_brightness(channel, 100)
time.sleep(2)
set_brightness(channel, 0)

