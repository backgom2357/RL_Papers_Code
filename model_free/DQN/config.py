class Config():
    def __init__(self):

        self.ATARI_GAMES = {
            0:'BreakoutDeterministic-v0',
            1:'BoxingDeterministic-v0',
            2:'MontezumaRevenge-ramDeterministic-v0',
            3:'SeaquestDeterministic-v0',
            4:'SpaceInvadersDeterministic-v0'
            }
        
        self.FRAME_SIZE = 84
        self.REPLAY_MEMORY_SIZE = 1000000
        self.AGENT_HISTORY_LENGHTH = 4
        self.LEARNING_RATE = 0.00025
        self.epsilon = 1
        self.BATCH_SIZE = 32
        self.FINAL_EXPLORATION = 0.1
        self.FINAL_EXPLORATION_FRAME = 1000000
        self.DISCOUNT_FACTOR = 0.99
        self.REPLAY_START_SIZE = 50000
        self.TARGET_NETWORK_UPDATE_FREQUENCY = 10000
        self.SKIP_FRAMES = 4
        self.NO_OPS = 30

        self.wandb = {'FRAME_SIZE':self.FRAME_SIZE}