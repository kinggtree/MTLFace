import torch
# import torch.distributed as dist
from models.mtlface import MTLFace

if __name__ == '__main__':
    parser = MTLFace.parser()
    opt = parser.parse_args()
    print(opt)

    # dist.init_process_group(backend='nccl', init_method='env://')
    torch.cuda.set_device(0)
    model = MTLFace(opt)
    model.fit()
