import time
import sys
import os

import pyautogui
import numpy as np


def run():
    pyautogui.click(1920 / 2, 1080 / 2)
    time.sleep(2.5)
    pyautogui.press('space')
    time.sleep(1)

while True: run()
