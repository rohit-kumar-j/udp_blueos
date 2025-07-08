#!/usr/bin/python3
import time, sys, signal, argparse, curses, threading
from queue import Queue, Empty
from pca9685 import PCA9685

NEUTRAL = 1500
FWD     = 1600
REV     = 2 * NEUTRAL - FWD
PWM_CH  = range(6)
LOOP_DT = 0.02
HOLD_GRACE = 0.15

pca = PCA9685()
pca.set_pwm_frequency(50)
pca.output_enable()

def all_neutral():
    for ch in PWM_CH:
        pca.pwm[ch] = NEUTRAL

def panic_exit(*_):
    all_neutral()
    try: curses.endwin()
    except: pass
    sys.exit(0)

signal.signal(signal.SIGINT, panic_exit)
signal.signal(signal.SIGHUP, panic_exit)

cmd_q = Queue()

def worker():
    while True:
        cmd = cmd_q.get()
        if cmd == 'STOP':
            panic_exit()
        elif cmd == 'NEU':
            all_neutral()
        elif cmd == 'W':
            pca.pwm[0]=pca.pwm[1]=FWD ; pca.pwm[2]=pca.pwm[3]=REV
        elif cmd == 'S':
            pca.pwm[0]=pca.pwm[1]=REV ; pca.pwm[2]=pca.pwm[3]=FWD
        elif cmd == 'A':
            pca.pwm[0]=pca.pwm[2]=FWD ; pca.pwm[1]=pca.pwm[3]=REV
        elif cmd == 'D':
            pca.pwm[0]=pca.pwm[2]=REV ; pca.pwm[1]=pca.pwm[3]=FWD
        elif cmd == 'Q':
            pca.pwm[4]=pca.pwm[5]=FWD
        elif cmd == 'E':
            pca.pwm[4]=pca.pwm[5]=REV

threading.Thread(target=worker, daemon=True).start()

def flush_queue(q):
    while not q.empty():
        try:
            q.get_nowait()
        except Empty:
            break

def curses_loop(stdscr):
    curses.cbreak()
    stdscr.nodelay(True)
    stdscr.addstr(0, 0, "Hold W/S/A/D/Q/E.  Press X or Ctrl-C to stop.\n")

    last_key_time = 0
    sent_neutral  = True

    while True:
        ch = stdscr.getch()
        now = time.time()

        if ch != -1:
            key = chr(ch).upper()
            if key == 'X':
                cmd_q.put('STOP')
            elif key in 'WSADQE':
                cmd_q.put(key)
                last_key_time = now
                sent_neutral  = False

        elif not sent_neutral and (now - last_key_time) > HOLD_GRACE:
            flush_queue(cmd_q)          # <- new line
            cmd_q.put('NEU')
            sent_neutral = True

        time.sleep(LOOP_DT)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--keys", action="store_true")
    ap.add_argument("--func1", action="store_true")
    ap.add_argument("--func2", action="store_true")
    ap.add_argument("--func3", action="store_true")
    args = ap.parse_args()

    if args.keys:
        curses.wrapper(curses_loop)
    elif args.func1:
        function_one()
    elif args.func2:
        function_two()
    elif args.func3:
        function_three()
    else:
        ap.print_help()

if __name__ == "__main__":
    main()

