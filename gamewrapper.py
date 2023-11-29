from pyboy.plugins.game_wrapper_super_mario_land import GameWrapperSuperMarioLand

ADDR_LIVES_LEFT = 0xDA15
ADDR_LIVES_LEFT_DISPLAY = 0x9806
ADDR_WORLD_LEVEL = 0xFFB4
ADDR_WIN_COUNT = 0xFF9A


def _bcm_to_dec(value):
    return (value >> 4) * 10 + (value & 0x0F)


def custom_post_tick(self):
    self._tile_cache_invalid = True
    self._sprite_cache_invalid = True

    world_level = self.pyboy.get_memory_value(ADDR_WORLD_LEVEL)
    self.world = world_level >> 4, world_level & 0x0F
    blank = 300
    self.coins = self._sum_number_on_screen(9, 1, 2, blank, -256)
    self.lives_left = _bcm_to_dec(self.pyboy.get_memory_value(ADDR_LIVES_LEFT))
    self.score = self._sum_number_on_screen(0, 1, 6, blank, -256)
    self.time_left = self._sum_number_on_screen(17, 1, 3, blank, -256)

    level_block = self.pyboy.get_memory_value(0xC0AB)
    mario_x = self.pyboy.get_memory_value(0xC202)
    scx = self.pyboy.botsupport_manager().screen().tilemap_position_list()[16][0]
    self.level_progress = level_block * 16 + (scx - 7) % 16 + mario_x

    if self.game_has_started:
        self._level_progress_max = max(self.level_progress, self._level_progress_max)
        # end_score = self.score + self.time_left * 10
        self.fitness = self.lives_left * 100 + self._level_progress_max


class CustomGameWrapper(GameWrapperSuperMarioLand):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def post_tick(self):
        super().post_tick()
        if self.game_has_started:
            # this is the default implementation of fitness
            # TODO replace with better fitness function
            self._level_progress_max = max(
                self.level_progress, self._level_progress_max
            )
            # end_score = self.score + self.time_left * 10
            self.fitness = self.lives_left * 100 + self._level_progress_max


class CustomFitness():
    def __init__(self, game_wrapper):
        self.game_wrapper = game_wrapper

    