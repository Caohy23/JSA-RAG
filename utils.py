import torch
import torch.distributed as dist


class Gather(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.tensor):
        output = [torch.zeros_like(x) for _ in range(dist.get_world_size())]
        dist.all_gather(output, x)
        return tuple(output)

    @staticmethod
    def backward(ctx, *grads):
        all_gradients = torch.stack(grads)
        dist.all_reduce(all_gradients)
        return all_gradients[dist.get_rank()]


def gather_wgrad(x: torch.tensor, dim: int = 0):
    if not dist.is_initialized():
        return x
    x_gather = Gather.apply(x)
    x_gather = torch.cat(x_gather, dim=dim)
    return x_gather

    

@torch.no_grad()
def all_gather(x: torch.tensor, dim: int = 0):
    if not dist.is_initialized():
        return x
    # gather x on all devices into x_gather
    x_gather = [torch.ones_like(x) for _ in range(dist.get_world_size())]
    dist.all_gather(x_gather, x)

    x_gather = torch.cat(x_gather, dim=dim)
    return x_gather


@torch.no_grad()
def varsize_all_gather(x: torch.Tensor, dim: int = 0):
    """all_gather tensors of different sizes along the specified dimension with concatenation"""
    if not dist.is_initialized():
        return x
    # first gather size
    size = x.size(dim)
    tensor_size = torch.tensor(size, device=x.device, dtype=torch.int64)
    all_sizes = [torch.zeros_like(tensor_size) for _ in range(dist.get_world_size())]
    dist.all_gather(all_sizes, tensor_size)
    max_size = max([s.item() for s in all_sizes])
    # padding then concatenation
    padding_tuple_size = [max_size - size if k == dim else x.size(k) for k in range(x.ndim)]
    tensor_tuple_size = [max_size if k == dim else x.size(k) for k in range(x.ndim)]
    if size != max_size:
        padding = torch.empty(size=padding_tuple_size, dtype=x.dtype, device=x.device)
        x = torch.cat((x, padding), dim=dim)

    tensor_list = [torch.empty(tensor_tuple_size, device=x.device, dtype=x.dtype) for s in all_sizes]

    dist.all_gather(tensor_list=tensor_list, tensor=x)
    # remove padding
    tensor_list = [torch.narrow(tensor, dim, start=0, length=all_sizes[k]) for k, tensor in enumerate(tensor_list)]
    output = torch.cat(tensor_list, dim=dim)
    return output

@torch.no_grad()
def get_varsize(x: torch.Tensor, dim: int = 0):
    """gather tensors of different sizes along the first dimension"""
    if not dist.is_initialized():
        return torch.tensor([x.size(dim)])

    # determine max size
    size = torch.tensor([x.size(dim)], device=x.device, dtype=torch.int)
    allsizes = [torch.zeros_like(size) for _ in range(dist.get_world_size())]
    dist.all_gather(allsizes, size)
    allsizes = torch.cat(allsizes)
    return allsizes



@torch.no_grad()
def varsize_gather(x: torch.Tensor, dst: int = 0, dim: int = 0):
    """gather tensors of different sizes along the specified dimension"""
    if not dist.is_initialized():
        return x

    size = x.size(dim)
    tensor_size = torch.tensor(size, device=x.device, dtype=torch.int64)
    all_sizes = [torch.zeros_like(tensor_size) for _ in range(dist.get_world_size())]
    dist.all_gather(all_sizes, tensor_size)
    max_size = max([s.item() for s in all_sizes])

    padding_tuple_size = [max_size - size if k == dim else x.size(k) for k in range(x.ndim)]
    tensor_tuple_size = [max_size if k == dim else x.size(k) for k in range(x.ndim)]
    if size != max_size:
        padding = torch.empty(size=padding_tuple_size, dtype=x.dtype, device=x.device)
        x = torch.cat((x, padding), dim=dim)

    if dist.get_rank() == dst:
        tensor_list = [torch.empty(tensor_tuple_size, device=x.device, dtype=x.dtype) for s in all_sizes]
    else:
        tensor_list = None

    dist.gather(x, gather_list=tensor_list, dst=dst)
    if dist.get_rank() == dst:
        tensor_list = [torch.narrow(tensor, dim, start=0, length=all_sizes[k]) for k, tensor in enumerate(tensor_list)]

    return tensor_list

