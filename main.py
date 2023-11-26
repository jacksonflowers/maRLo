import sys
from pyboy import PyBoy, WindowEvent
from pyboy.plugins.game_wrapper_super_mario_land import GameWrapperSuperMarioLand

from gamewrapper import custom_post_tick


if __name__ == "__main__":
    filename = "./roms/mario.gb"
    quiet = "--quiet" in sys.argv
    pyboy = PyBoy(
        filename,
        window_type="headless" if quiet else "SDL2",
        window_scale=3,
        debug=not quiet,
        game_wrapper=True,
    )

    pyboy.set_emulation_speed(0)
    assert pyboy.cartridge_title() == "SUPER MARIOLAN"

    mario: GameWrapperSuperMarioLand = pyboy.game_wrapper()
    mario.post_tick = custom_post_tick
    mario.start_game()

    last_fitness = 0

    # pyboy.send_input(WindowEvent.PRESS_ARROW_RIGHT)
    while not pyboy.tick():
        pass
        # assert mario.fitness >= last_fitness
        # last_fitness = mario.fitness

        # pyboy.tick()
        # if mario.lives_left == 1:
        #     assert last_fitness == 27700
        #     assert (
        #         mario.fitness == 17700
        #     )  # Loosing a live, means 10.000 points in this fitness scoring
        #     print(mario)
        #     break
    else:
        print("Mario didn't die?")
        exit(2)

    mario.reset_game()
    assert mario.lives_left == 2

    pyboy.stop()
