import os
import sys

from pyboy import PyBoy, WindowEvent
import numpy as np
from PIL import Image

# Makes us able to import PyBoy from the directory below
file_path = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, file_path + "/..")

# Check if the ROM is given through argv
if len(sys.argv) > 1:
    filename = sys.argv[1]
else:
    print("Usage: python mario_boiler_plate.py [ROM file]")
    exit(1)

quiet = "--quiet" in sys.argv
pyboy = PyBoy(filename, window_type="headless" if quiet else "SDL2", window_scale=3, debug=False, game_wrapper=True)
pyboy.set_emulation_speed(0)
assert pyboy.cartridge_title() == "SUPER MARIOLAN"

mario = pyboy.game_wrapper()
mario.start_game()

assert mario.score == 0
assert mario.lives_left == 2
assert mario.time_left == 400
assert mario.world == (1, 1)
assert mario.fitness == 0 # A built-in fitness score for AI development
last_fitness = 0

img = Image.fromarray(np.asarray(pyboy.botsupport_manager().screen().screen_ndarray())[15:,:,:], 'RGB')
img.save('yo.png')
# while True:
#     pyboy.tick()
#     print(mario.level_progress)
#     if mario.lives_left == 1:
#         pyboy.stop()

mario.reset_game()
assert mario.lives_left == 2

pyboy.stop()