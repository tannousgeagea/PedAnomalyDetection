class Config:
    
    def __init__(self):
        
        self.modelName = 'svdd'
        self.source = '../datasets/dataset_AoA_change/'
        self.batch_size = 16
        self.config = [
            [32, 3, 2],
            [64, 3, 2],
            [128, 3, 2],
        ]
        
        
        self.latent_dim = 16
        self.lr=0.0001
        self.epochs=300 
        self.objective='soft-boundry' 
        self.ae_epochs= 20
        self.nu=0.02
        self.gamma=0.1
        self.Normal= ['Walking', 'Running', 'Jogging', 'TalkOnThePhone']
        
        self.test_seed = [42, 123, 122, 2023, 145, 100, 22, 147, 95, 32]
        self.seed = 147