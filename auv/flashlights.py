import time
from pca9685 import PCA9685

# Global PCA instance (singleton)
pca = None

def setup_led(channel, freq=50):
    """
    Initialize PCA9685, enable output, set frequency.
    Returns the channel number for convenience.
    """
    global pca
    if pca is None:
        pca = PCA9685()
        pca.set_pwm_frequency(freq)
        pca.output_enable()
    # Initialize channel to off
    pca.pwm[channel] = 0
    return channel

def set_brightness(channel, brightness_percent):
    """
    Set brightness 0-100% on given channel.
    Maps 0-100% to 0-4095 duty cycle counts internally.
    """
    brightness = max(0, min(100, brightness_percent))
    counts = int((brightness / 100) * 4095)
    pca.pwm[channel] = counts

def us_to_counts(pulse_us, freq=50):
    """
    Convert servo pulse width in microseconds to PCA9685 counts (0-4095).
    """
    period_us = 1_000_000 / freq  # period in microseconds
    counts = int((pulse_us / period_us) * 4095)
    return counts

