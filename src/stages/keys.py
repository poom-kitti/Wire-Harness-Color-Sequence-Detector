"""This module contains the acceptable keys for each stage."""
DEFAULT_WAIT_KEY_TIME = int(1000 / 120)  # ms

N_KEY = ord("n")
R_KEY = ord("r")
Q_KEY = ord("q")
Y_KEY = ord("y")
ENTER_KEY = 13
ESC_KEY = 27

QUIT_KEYS = (Q_KEY, ESC_KEY)
ACCEPTABLE_KEYS_FOR_INITIALIZE_STAGE = (Y_KEY, N_KEY, Q_KEY, ESC_KEY)
ACCEPTABLE_KEYS_FOR_ACCEPT_CAPTURED_BG_FRAME = (Y_KEY, N_KEY, Q_KEY, ESC_KEY)
ACCEPTABLE_KEYS_FOR_ACCEPT_CAPTURED_REFERENCE_WIRE_HOUSING = (Y_KEY, N_KEY, Q_KEY, ESC_KEY)
ACCEPTABLE_KEYS_FOR_CHECK_CAPTURE_STAGE = (N_KEY, R_KEY, Q_KEY, ESC_KEY)
