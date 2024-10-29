import numpy as np

class dummy_model:
    def __init__(self, latent_size=1024, random=False):
        self.latent_size = latent_size
        self.random=random
    
    def __call__(self, tensor):

        if len(np.shape(tensor)) == 3:
            tensor = tensor[np.newaxis,:,:,:]
        elif len(np.shape(tensor))!=4:
            raise TypeError('Incompatible shape for input tensor')
    
        out_tensor = np.array([]).reshape([0,self.latent_size])
        for i in range(np.shape(tensor)[0]):
            if self.random:
                self.latent_vector = np.random.rand(1,self.latent_size)
            else:
                self.latent_vector = np.ones([1,self.latent_size])
            out_tensor = np.vstack((out_tensor, self.latent_vector))

        return out_tensor
    

#model = dummy_model()
#fake_input = np.random.rand(512,512,3)
#result = model(fake_input)
#print(np.shape(result))
#print(result)