import time
from pca9685 import PCA9685

# Coinstants
PWM_CHANNEL =  8

# Setup PCA9685 and send signal
pca = PCA9685()
pca.set_pwm_frequency(50)  # 50Hz for servo
pca.output_enable()
pca.pwm[PWM_CHANNEL] = 1900

# Wait for 2 seconds
time.sleep(2)

pca.pwm[PWM_CHANNEL] = 1500
