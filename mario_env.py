from gymnasium import Env
from gymnasium.spaces import Discrete, MultiDiscrete, Box
from pyboy import PyBoy, WindowEvent
from pyboy.botsupport.constants import TILES
import numpy as np

# TODO:
# skip frames is not implemented well (observations are also skip_frames frames apart)
# continue from checkpoint
# monitoring

class MarioEnv(Env):
    def __init__(self, args):
        self.metadata = {'render_fps': 60}
        self.render_mode = 'rgb_array'

        self.pyboy = PyBoy(args.gb_path, game_wrapper=True, window_type='headless')
        assert self.pyboy.cartridge_title() == 'SUPER MARIOLAN'

        self.skip_frames = 2

        self.game_wrapper = self.pyboy.game_wrapper()
        self.last_fitness = self.compute_fitness()

        self._DO_NOTHING = WindowEvent.PASS
        self._buttons = [
            WindowEvent.PRESS_ARROW_UP, WindowEvent.PRESS_ARROW_DOWN, WindowEvent.PRESS_ARROW_RIGHT,
            WindowEvent.PRESS_ARROW_LEFT, WindowEvent.PRESS_BUTTON_A,
        ]
        # self._buttons = [
        #     WindowEvent.PRESS_ARROW_UP, WindowEvent.PRESS_ARROW_RIGHT, WindowEvent.PRESS_BUTTON_A,
        # ]
        self._button_is_pressed = {button: False for button in self._buttons}

        self._buttons_release = [
            WindowEvent.RELEASE_ARROW_UP, WindowEvent.RELEASE_ARROW_DOWN, WindowEvent.RELEASE_ARROW_RIGHT,
            WindowEvent.RELEASE_ARROW_LEFT, WindowEvent.RELEASE_BUTTON_A,
        ]
        # self._buttons_release = [
        #     WindowEvent.RELEASE_ARROW_UP, WindowEvent.RELEASE_ARROW_RIGHT, WindowEvent.RELEASE_BUTTON_A,
        # ]
        self._release_button = {button: r_button for button, r_button in zip(self._buttons, self._buttons_release)}

        self.actions = [self._DO_NOTHING] + self._buttons
        if args.action_type == "all":
            self.actions += self._buttons_release
        elif args.action_type not in ["press", "toggle"]:
            raise ValueError(f"action_type {args.action_type} is invalid")
        self.action_type = args.action_type
        # simultaneous actions?

        self.action_space = Discrete(len(self.actions))

        if args.observation_type == "raw":
            screen = np.asarray(self.pyboy.botsupport_manager().screen().screen_ndarray())  # (144, 160, 3)
            self.observation_space = Box(low=0, high=255, shape=screen.shape, dtype=np.uint8)
        elif args.observation_type in ["tiles", "compressed", "minimal"]:
            size_ids = TILES
            if args.observation_type == "compressed":
                try:
                    size_ids = np.max(self.game_wrapper.tiles_compressed) + 1
                except AttributeError:
                    raise AttributeError(
                        "You need to add the tiles_compressed attibute to the game_wrapper to use the compressed observation_type"
                    )
            elif args.observation_type == "minimal":
                try:
                    size_ids = np.max(self.game_wrapper.tiles_minimal) + 1
                except AttributeError:
                    raise AttributeError(
                        "You need to add the tiles_minimal attibute to the game_wrapper to use the minimal observation_type"
                    )
            # nvec = size_ids * np.ones(self.game_wrapper.shape)
            # self.observation_space = MultiDiscrete(nvec)
            self.observation_space = Box(low=0, high=255, shape=self.game_wrapper.shape, dtype=np.uint8)
        else:
            raise NotImplementedError(f"observation_type {args.observation_type} is invalid")
        self.observation_type = args.observation_type

        self._started = False
        
    def _get_observation(self):
        if self.observation_type == "raw":
            observation = np.asarray(self.pyboy.botsupport_manager().screen().screen_ndarray(), dtype=np.uint8)
        elif self.observation_type in ["tiles", "compressed", "minimal"]:
            observation = self.game_wrapper._game_area_np(self.observation_type)
        else:
            raise NotImplementedError(f"observation_type {self.observation_type} is invalid")
        return observation
        
    def step(self, action_id):
        action = self.actions[action_id]
        if action == self._DO_NOTHING:
            for _ in range(self.skip_frames):
                pyboy_done = self.pyboy.tick()
        else:
            if self.action_type == "toggle":
                if self._button_is_pressed[action]:
                    self._button_is_pressed[action] = False
                    action = self._release_button[action]
                else:
                    self._button_is_pressed[action] = True

            self.pyboy.send_input(action)
            for _ in range(self.skip_frames):
                pyboy_done = self.pyboy.tick()

            if self.action_type == "press":
                self.pyboy.send_input(self._release_button[action])
        
        final_fitness = self.last_fitness

        new_fitness = self.compute_fitness()
        reward = new_fitness - self.last_fitness
        self.last_fitness = new_fitness
        
        observation = self._get_observation()
        done = pyboy_done or self.game_wrapper.game_over()

        world = self.game_wrapper.world
        level_progress = self.game_wrapper.level_progress

        return observation, reward, False, done, {'final_fitness': final_fitness, 'world': world, 'level_progress': level_progress}
    
    def reset(self, seed=None, options=None):
        final_fitness = self.last_fitness
        if not self._started:
            self.game_wrapper.start_game()
            self._started = True
        else:
            self.game_wrapper.reset_game()
        self.last_fitness = self.compute_fitness()
        self.button_is_pressed = {button: False for button in self._buttons}
        return self._get_observation(), {'final_fitness': final_fitness}

    def render(self):
        return np.array(self.pyboy.screen_image())

    def close(self):
        self.pyboy.stop(save=False)

    def compute_fitness(self):
        return self.game_wrapper.lives_left * 20 + sum(self.game_wrapper.world) * 50 + self.game_wrapper.level_progress + self.game_wrapper.time_left
        # return self.game_wrapper.fitness
