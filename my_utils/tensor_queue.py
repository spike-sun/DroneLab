import torch

class TensorQueue:
    def __init__(self, device, batch_size, maxlen, *shape):
        self.device = device
        self.batch_size = batch_size
        self.maxlen = maxlen
        self.shape = shape
        self.buffer = torch.zeros((batch_size, maxlen) + shape, device=device)
    
    def init(self, data: torch.Tensor):
        self.buffer[:,:] = data.unsqueeze(1)
    
    def init_ids(self, ids, data: torch.Tensor):
        self.buffer[ids,:] = data[ids].unsqueeze(1)
    
    def append(self, data: torch.Tensor):
        self.buffer = torch.roll(self.buffer, shifts=-1, dims=1)
        self.buffer[:,-1] = data
    
    def newest(self):
        return self.buffer[:,-1]

    def oldest(self):
        return self.buffer[:,0]
    
    def __str__(self):
        return self.buffer.__str__()


if __name__ == '__main__':
    q = TensorQueue('cuda:0', 1, 3, 2)
    q.init(torch.tensor([[1,2]]))
    print(q)
    q.append(torch.tensor([[3,4]]))
    print(q)
    q.append(torch.tensor([[5,6]]))
    print(q)
    q.append(torch.tensor([[7,8]]))
    print(q)
    print(q.oldest())
    print(q.newest())
    q.init_ids([0], torch.tensor([[9,10]]))
    print(q)