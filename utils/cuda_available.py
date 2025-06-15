import torch

def cuda_available():
    cuda_available = torch.cuda.is_available()
    print(torch.__version__)
    if cuda_available:
    # 打印 CUDA 设备信息
        print("CUDA 可用！")
        print("GPU 设备数量:", torch.cuda.device_count())
        print("当前使用的 GPU 设备:", torch.cuda.current_device())
        print("设备名称:", torch.cuda.get_device_name(torch.cuda.current_device()))
    else:
        print("CUDA 不可用，将使用 CPU 运行 PyTorch。")

    # 设置默认设备为 CUDA，如果 CUDA 可用
    device = torch.device('cuda' if cuda_available else 'cpu')

    # 示例：在 GPU 上创建一个 Tensor
    if cuda_available:
    # 在 GPU 上创建一个 Tensor
        tensor_on_gpu = torch.rand(3, 3).to(device)
        print("在 GPU 上创建的 Tensor:", tensor_on_gpu)
    else:
        print("没有可用的 GPU，无法在 GPU 上创建 Tensor。")