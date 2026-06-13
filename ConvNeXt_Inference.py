import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import numpy as np
import softposit as sp
import os
from multiprocessing import Pool, cpu_count

# Note: The architecture blocks must be duplicated here so the inference
# script can operate as a completely standalone execution file.

# ==========================================
# 1. NATIVE FORMAT CASTER
# ==========================================
def cast_softposit(val, mode):
    try:
        if mode == 'POSIT32_2': return float(sp.posit32(val))
        if mode == 'POSIT16_1': return float(sp.posit16(val))
        if mode == 'POSIT8_0':  return float(sp.posit8(val))
    except Exception:
        return 0.0
    return val

vec_cast_softposit = np.vectorize(cast_softposit)

def process_chunk(chunk_data):
    chunk, mode = chunk_data
    return vec_cast_softposit(chunk, mode)

def apply_format(tensor, mode):
    if mode == 'FP32': return tensor
    elif mode == 'FP16': return tensor.to(torch.float16).to(torch.float32)
    elif mode == 'FXP32': 
        scale = 2**16
        return torch.clamp(torch.round(tensor * scale) / scale, min=-32768.0, max=32767.999)
    elif mode == 'FXP16': 
        scale = 2**8
        return torch.clamp(torch.round(tensor * scale) / scale, min=-128.0, max=127.996)
    elif mode.startswith('POSIT'):
        device = tensor.device
        np_arr = tensor.detach().cpu().numpy()
        flat_arr = np_arr.flatten()
        
        num_workers = max(1, cpu_count() - 2) 
        chunks = np.array_split(flat_arr, num_workers)
        chunk_args = [(chunk, mode) for chunk in chunks]
        
        with Pool(processes=num_workers) as pool:
            results = pool.map(process_chunk, chunk_args)
            
        quantized_np = np.concatenate(results).reshape(np_arr.shape)
        return torch.from_numpy(quantized_np).to(device).float()
    return tensor

# ==========================================
# 2. PURE FORMAT AUTOGRAD 
# ==========================================
class PureFormatConv2dFunc(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input_t, weight_t, bias_t, stride, padding, groups, mode):
        f_input = apply_format(input_t, mode)
        f_weight = apply_format(weight_t, mode)
        f_bias = apply_format(bias_t, mode) if bias_t is not None else None
        out = F.conv2d(f_input, f_weight, f_bias, stride=stride, padding=padding, groups=groups)
        return apply_format(out, mode)

class PureFormatLinearFunc(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input_t, weight_t, bias_t, mode):
        f_input = apply_format(input_t, mode)
        f_weight = apply_format(weight_t, mode)
        f_bias = apply_format(bias_t, mode) if bias_t is not None else None
        out = F.linear(f_input, f_weight, f_bias)
        return apply_format(out, mode)

# ==========================================
# 3. HARDWARE LAYERS
# ==========================================
class HardwareConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=1, groups=1, mode='FP32'):
        super().__init__()
        self.mode = mode
        self.stride = stride
        self.padding = padding
        self.groups = groups
        self.weight = nn.Parameter(torch.empty(out_channels, in_channels // groups, kernel_size, kernel_size))
        self.bias = nn.Parameter(torch.zeros(out_channels))

    def forward(self, x):
        return PureFormatConv2dFunc.apply(x, self.weight, self.bias, self.stride, self.padding, self.groups, self.mode)

class HardwareLinear(nn.Module):
    def __init__(self, in_features, out_features, mode='FP32'):
        super().__init__()
        self.mode = mode
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias = nn.Parameter(torch.zeros(out_features))

    def forward(self, x):
        return PureFormatLinearFunc.apply(x, self.weight, self.bias, self.mode)

class HardwareLayerNormChannelsFirst(nn.Module):
    def __init__(self, normalized_shape, eps=1e-6, mode='FP32'):
        super().__init__()
        self.mode = mode
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))

    def forward(self, x):
        f_x = apply_format(x, self.mode)
        u = f_x.mean(1, keepdim=True)
        s = (f_x - u).pow(2).mean(1, keepdim=True)
        normed = (f_x - u) / torch.sqrt(s + self.eps)
        f_weight = apply_format(self.weight[:, None, None], self.mode)
        f_bias = apply_format(self.bias[:, None, None], self.mode)
        return apply_format(f_weight * normed + f_bias, self.mode)

# ==========================================
# 4. CONVNEXT ARCHITECTURE
# ==========================================
class HardwareConvNeXtBlock(nn.Module):
    def __init__(self, dim, mode='FP32'):
        super().__init__()
        self.mode = mode
        self.dwconv = HardwareConv2d(dim, dim, kernel_size=7, stride=1, padding=3, groups=dim, mode=mode)
        self.norm_weight = nn.Parameter(torch.ones(dim))
        self.norm_bias = nn.Parameter(torch.zeros(dim))
        self.pwconv1 = HardwareLinear(dim, 4 * dim, mode=mode)
        self.act = nn.GELU()
        self.pwconv2 = HardwareLinear(4 * dim, dim, mode=mode)
        self.gamma = nn.Parameter(1e-6 * torch.ones((dim)))

    def forward(self, x):
        input_tensor = apply_format(x, self.mode)
        x = self.dwconv(input_tensor)
        x = x.permute(0, 2, 3, 1) 
        
        f_x = apply_format(x, self.mode)
        u = f_x.mean(-1, keepdim=True)
        s = (f_x - u).pow(2).mean(-1, keepdim=True)
        normed = (f_x - u) / torch.sqrt(s + 1e-6)
        f_nw = apply_format(self.norm_weight, self.mode)
        f_nb = apply_format(self.norm_bias, self.mode)
        x = apply_format(f_nw * normed + f_nb, self.mode)
        
        x = self.pwconv1(x)
        x = apply_format(self.act(x), self.mode)
        x = self.pwconv2(x)
        
        f_gamma = apply_format(self.gamma, self.mode)
        x = apply_format(f_gamma * x, self.mode)
        
        x = x.permute(0, 3, 1, 2) 
        return apply_format(input_tensor + x, self.mode)

class ConvNeXt_Hardware(nn.Module):
    def __init__(self, mode, num_classes=10, depths=[3, 3, 9, 3], dims=[96, 192, 384, 768]):
        super().__init__()
        self.mode = mode
        self.downsample_layers = nn.ModuleList() 
        stem = nn.Sequential(
            HardwareConv2d(3, dims[0], kernel_size=2, stride=2, padding=0, mode=mode),
            HardwareLayerNormChannelsFirst(dims[0], mode=mode)
        )
        self.downsample_layers.append(stem)
        
        for i in range(3):
            self.downsample_layers.append(nn.Sequential(
                HardwareLayerNormChannelsFirst(dims[i], mode=mode),
                HardwareConv2d(dims[i], dims[i+1], kernel_size=2, stride=2, padding=0, mode=mode),
            ))

        self.stages = nn.ModuleList() 
        for i in range(4):
            self.stages.append(nn.Sequential(*[HardwareConvNeXtBlock(dim=dims[i], mode=mode) for _ in range(depths[i])]))
            
        self.norm_weight = nn.Parameter(torch.ones(dims[-1]))
        self.norm_bias = nn.Parameter(torch.zeros(dims[-1]))
        self.head = HardwareLinear(dims[-1], num_classes, mode=mode)

    def forward(self, x):
        x = apply_format(x, self.mode)
        for i in range(4):
            x = self.downsample_layers[i](x)
            x = self.stages[i](x)
            
        x = x.mean([-2, -1])
        f_x = apply_format(x, self.mode)
        u = f_x.mean(-1, keepdim=True)
        s = (f_x - u).pow(2).mean(-1, keepdim=True)
        normed = (f_x - u) / torch.sqrt(s + 1e-6)
        
        f_nw = apply_format(self.norm_weight, self.mode)
        f_nb = apply_format(self.norm_bias, self.mode)
        x = apply_format(f_nw * normed + f_nb, self.mode)
        
        return self.head(x)

# ==========================================
# 5. MASTER INFERENCE LOOP
# ==========================================
if __name__ == "__main__":
    MODES = ['FP32', 'FP16', 'FXP32', 'FXP16', 'POSIT32_2', 'POSIT16_1', 'POSIT8_0']
    
    transform = transforms.Compose([
        transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=False, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=False)

    for mode in MODES:
        print(f"\n{'='*50}")
        print(f"EVALUATING NATIVE FORMAT: {mode}")
        print(f"{'='*50}")
        
        weight_file = f"convnext_native_weights_{mode}.pth"
        if not os.path.exists(weight_file):
            print(f"SKIPPING: Weights file {weight_file} not found. Ensure training finished.")
            continue

        if 'POSIT' in mode:
            device = torch.device('cpu')
        else:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        model = ConvNeXt_Hardware(mode=mode).to(device)
        model.load_state_dict(torch.load(weight_file, map_location=device), strict=True)
        model.eval()

        top1_correct = 0
        top5_correct = 0
        total = 0

        with torch.no_grad():
            for i, (inputs, labels) in enumerate(testloader):
                inputs, labels = inputs.to(device), labels.to(device)
                
                # Format raw image pixels
                inputs = apply_format(inputs, mode)
                outputs = model(inputs)
                
                _, pred = outputs.topk(5, 1, True, True)
                pred = pred.t()
                correct = pred.eq(labels.view(1, -1).expand_as(pred))

                top1_correct += correct[0].sum().item()
                top5_correct += correct[:5].sum().item()
                total += labels.size(0)

                if (i+1) % 40 == 0:
                    print(f"[{mode}] Processed {total}/10000 test images...")

        top1_acc = 100 * top1_correct / total
        top5_acc = 100 * top5_correct / total

        print(f"\nRESULTS FOR {mode}:")
        print(f"Top-1 Accuracy: {top1_acc:.2f}%")
        print(f"Top-5 Accuracy: {top5_acc:.2f}%")
