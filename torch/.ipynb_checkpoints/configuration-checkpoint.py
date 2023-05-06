class Configuration:
    
    def __init__(self):
        
        self.modelName = 'ntl'
        self.source = '../datasets/dataset_AoA_change/'
        self.batch_size = 16
        self.config_file = 'NeuTraLAD/config/config.yml'
        
        self.latent_dim = 16
        self.lr=0.0001
        self.epochs=20 
        self.nu=0.02
        self.gamma=0.1
        self.Normal= ['Walking', 'Running', 'Jogging', 'TalkingOnThePhone']
        
        self.radius = 30
        self.lamda = 1
        self.optim_idx = 1
        
        
        self.test_seed = [42, 123, 122, 2023, 145, 100, 22, 147, 95, 32]
        self.seed = 147