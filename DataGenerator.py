import SimpleITK as sitk
import numpy as np

class DataGenerator:
    def __init__(self,path_X,path_Y,batch_size,callback_img_num,test_size=0.9):
        
        self.X = self.get_img_array(path_X)
        self.Y = self.get_img_array(path_Y)
        
        self.test_size = test_size
        self.batch_size = batch_size
        self.nb_elements = self.X.shape[0]
        self.train_idx, self.val_idx = self.get_index()
        
        self.x_callback = self.get_callback_array(self.X,callback_img_num)
        self.y_callback = self.get_callback_array(self.Y,callback_img_num)
        
    def split_data(self,index):
        
        limit_sample = np.floor(self.nb_elements * self.test_size).astype(int)
        val_idx = index[limit_sample:]
        train_idx = index[:limit_sample]
        
        self.train_idx_bkup = train_idx.copy()
        self.val_idx_bkup = val_idx.copy()
        
        return train_idx, val_idx
        
        
    def get_callback_array(self,tensor,img_num):
        
        idx = np.random.randint(0,tensor.shape[0],img_num)
        callback_array = tensor[idx,...,np.newaxis]
        
        return callback_array

    def get_img_array(self,path):
        
        image = np.load(path)
                
        return image

    def get_index(self):

        index = np.arange(self.X.shape[0])
        np.random.shuffle(index)
        train_idx, val_idx = self.split_data(index)
        
        return list(train_idx), list(val_idx)
    
    def num_step_epochs(self,train_val):
        
        if train_val == 'train':
            return np.ceil(len(self.train_idx_bkup) / self.batch_size).astype(int)
        else:
            return np.ceil(len(self.val_idx_bkup) / self.batch_size).astype(int)
    
    def epochs_gen(self,train_val):
        
        if train_val == 'train':
            self.train_idx = self.train_idx_bkup.copy()
            np.random.shuffle(self.train_idx)
            self.train_idx = list(self.train_idx)
        else:
            self.val_idx = self.val_idx_bkup.copy()
            np.random.shuffle(self.val_idx)
            self.val_idx = list(self.val_idx)
        
    def get_batch(self,train_val):
        
        if train_val == 'train':
            idxs = self.train_idx
        else:
            idxs = self.val_idx
            
        index = []
        if len(idxs) >= self.batch_size:
            for n in range(self.batch_size):
                index.append(idxs.pop())
            if len(idxs) == 0:
                self.epochs_gen(train_val)
        else:
            index = idxs
            self.epochs_gen(train_val)
        
        index = np.array(index)
        batch_X = self.X[index,...,np.newaxis]
        batch_Y = self.Y[index,...,np.newaxis]

        
        return batch_X, batch_Y
        
    def gen_train_batch(self):
        while True:
            X, Y = self.get_batch('train')
            yield (X, Y)
            
    def gen_val_batch(self):
        while True:
            X, Y = self.get_batch('val')
            yield (X, Y)
