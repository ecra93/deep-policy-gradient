import cv2

def process(state):

    # crop out bottom of the pacman bar
    state = state[1:171, :]

    # convert to grayscale, then to binary
    state = cv2.cvtColor(state, cv2.COLOR_BGR2GRAY)
    state = cv2.threshold(state, 100, 255, cv2.THRESH_BINARY)[1]

    # resize to 84,84
    state = cv2.resize(state, (84,84))

    return state
