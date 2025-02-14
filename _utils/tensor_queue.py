import torch

class TensorQueue:
    def __init__(self, shape, batch_size, maxlen, dtype, device='cpu'):
        self.shape = torch.Size(shape)
        self.batch_size = batch_size
        self.buffer = torch.zeros((batch_size, maxlen) + self.shape, dtype=dtype, device=device, requires_grad=False)  # (B, T, *shape)
    
    def init(self, data: torch.Tensor, ids=None):
        assert data.shape == (self.batch_size,) + self.shape
        if ids is None:
            self.buffer[:,:] = data.unsqueeze(1)
        else:
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
    q = TensorQueue(shape=[2], batch_size=1, maxlen=3, dtype=torch.long, device='cpu')
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
    q.init(torch.tensor([[9,10]]), [0])
    print(q)