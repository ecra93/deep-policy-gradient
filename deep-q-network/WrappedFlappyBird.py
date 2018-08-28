from ple import PLE
from ple.games.flappybird import FlappyBird

class WrappedFlappyBird():

    def __init__(self):
        self.score_counter = 0
        self.game = FlappyBird()
        self.env = PLE(self.game, fps=30, display_screen=True)

    def frame_step(self, action_vector):
        if action_vector[0] == 1:
            self.env.act(119)
        elif action_vector[1] == 1:
            self.env.act(1)

        frame = self.env.getScreenRGB()
        reward = self.get_action_reward()
        game_over = self.game.game_over()

        if game_over:
            self.game.reset()

        return frame, reward, game_over

    def get_action_reward(self):
        if self.game.game_over():
            self.score_counter = 0
            return -1
        elif self.score_counter < self.game.getScore():
            self.score_counter = self.game.getScore()
            return 1
        else:
            return 0.1
