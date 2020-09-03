class Config():
    def __init__(self):

        self.ATARI_GAMES = {
            0:'BreakoutDeterministic-v0',
            1:'BoxingDeterministic-v0',
            2:'MontezumaRevenge-ramDeterministic-v0',
            3:'SeaquestDeterministic-v0',
            4:'SpaceInvadersDeterministic-v0'
            }
        
        self.AGENT_HISTORY_LENGHTH = 4
        self.BATCH_SIZE = 32
        self.DISCOUNT_FACTOR = 0.99
        self.FINAL_EXPLORATION = 0.1
        self.FINAL_EXPLORATION_FRAME = 1000000
        self.FRAME_SIZE = 84
        self.LEARNING_RATE = 0.00025
        self.NO_OPS = 30
        self.REPLAY_MEMORY_SIZE = 1000000
        self.REPLAY_START_SIZE = 50000
        self.SKIP_FRAMES = 4
        self.TARGET_NETWORK_UPDATE_FREQUENCY = 10000
        
        self.epsilon = 1
        

        self.WANDB = {
            'AGENT_HISTORY_LENGHTH', AGENT_HISTORY_LENGHTH,
            'BATCH_SIZE' : BATCH_SIZE, 
            'DISCOUNT_FACTOR' : DISCOUNT_FACTOR, 
            'FINAL_EXPLORATION' : FINAL_EXPLORATION,
            'FINAL_EXPLORATION_FRAME' : FINAL_EXPLORATION_FRAME,
            'FRAME_SIZE' : FRAME_SIZE, 
            'LEARNING_RATE' : LEARNING_RATE,
            'NO_OPS' : NO_OPS, 
            'REPLAY_MEMORY_SIZE' : REPLAY_MEMORY_SIZE,
            'REPLAY_START_SIZE' : REPLAY_START_SIZE,
            'SKIP_FRAMES' : SKIP_FRAMES,
            'TARGET_NETWORK_UPDATE_FREQUENCY' : TARGET_NETWORK_UPDATE_FREQUENCY,
            }