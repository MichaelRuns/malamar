import time
import pyautogui

pyautogui.PAUSE = 0
pyautogui.FAILSAFE = False

KEY = {
    "UP": "w",
    "DOWN": "s",
    "LEFT": "a",
    "RIGHT": "d",
    "A": "j",
    "B": "k",
    "X": "u",
    "Y": "i",
    "SELECT": "enter",
    "START": "]",
    "L": "q",
    "R": "e"
}

ACTION_MAP = {
    "MOVE_1": ["A", "A"],
    "MOVE_2": ["DOWN", "A", "A"],
    "MOVE_3": ["DOWN", "DOWN", "A", "A"],
    "MOVE_4": ["DOWN", "DOWN", "DOWN", "A", "A"],

    "SWITCH_1": ["RIGHT", "A", "A"],
    "SWITCH_2": ["RIGHT", "A", "DOWN", "A"],
    "SWITCH_3": ["RIGHT", "A", "DOWN", "DOWN", "A"],
    "SWITCH_4": ["RIGHT", "A", "DOWN", "DOWN", "DOWN", "A"],
    "WALK_F": ["UP"]
}

PRESS = 0.05
GAP = 0.10

def press(key):
    pyautogui.keyDown(key)
    time.sleep(PRESS)
    pyautogui.keyUp(key)
    time.sleep(GAP)

def execute_action(action: str):
    for k in ACTION_MAP[action]:
        press(KEY[k] if k in KEY else k)

def main():
    time.sleep(3)  # gives you time to focus melonDS
    print('start')
    execute_action("WALK_F")
    print('end')

if __name__ == "__main__":
    main()
