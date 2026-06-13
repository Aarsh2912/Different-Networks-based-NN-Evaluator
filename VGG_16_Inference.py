import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import numpy as np
import softposit as sp
import os
from multiprocessing import Pool, cpu_count

# ==========================================
# 1. NATIVE FORMAT CASTER (MULTIPROCESSING)
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
    """Forces the tensor into the exact mathematical format boundaries."""
    if mode == 'FP32':
        return tensor
    elif mode == 'FP16':
        return tensor.to(torch.float16).to(torch.float32)
    elif mode == 'FXP32': # Q16.16
        scale = 2**16
        return torch.clamp(torch.round(tensor * scale) / scale, min=-32768.0, max=32767.999)
    elif mode == 'FXP16': # Q8.8
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
# 2. PURE FORMAT AUTOGRAD (FORWARD PASS)
# ==========================================
class PureFormatConv2dFunc(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input_t, weight_t, bias_t, stride, padding, mode):
        f_input = apply_format(input_t, mode)
        f_weight = apply_format(weight_t, mode)
        f_bias = apply_format(bias_t, mode) if bias_t is not None else None

        out = F.conv2d(f_input, f_weight, f_bias, stride, padding)
        f_out = apply_format(out, mode)

        ctx.save_for_backward(f_input, f_weight, f_bias)
        ctx.stride = stride
        ctx.padding = padding
        ctx.mode = mode
        return f_out

    @staticmethod
    def backward(ctx, grad_output):
        f_input, f_weight, f_bias = ctx.saved_tensors
        mode = ctx.mode

        f_grad_output = apply_format(grad_output, mode)

        grad_input = torch.nn.grad.conv2d_input(f_input.shape, f_weight, f_grad_output, stride=ctx.stride, padding=ctx.padding)
        grad_weight = torch.nn.grad.conv2d_weight(f_input, f_weight.shape, f_grad_output, stride=ctx.stride, padding=ctx.padding)
        
        f_grad_input = apply_format(grad_input, mode)
        f_grad_weight = apply_format(grad_weight, mode)
        f_grad_bias = apply_format(f_grad_output.sum(dim=(0, 2, 3)), mode) if f_bias is not None else None

        return f_grad_input, f_grad_weight, f_grad_bias, None, None, None

class PureFormatLinearFunc(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input_t, weight_t, bias_t, mode):
        f_input = apply_format(input_t, mode)
        f_weight = apply_format(weight_t, mode)
        f_bias = apply_format(bias_t, mode) if bias_t is not None else None

        out = F.linear(f_input, f_weight, f_bias)
        f_out = apply_format(out, mode)

        ctx.save_for_backward(f_input, f_weight, f_bias)
        ctx.mode = mode
        return f_out

    @staticmethod
    def backward(ctx, grad_output):
        f_input, f_weight, f_bias = ctx.saved_tensors
        mode = ctx.mode

        f_grad_output = apply_format(grad_output, mode)

        grad_input = f_grad_output.mm(f_weight)
        grad_weight = f_grad_output.t().mm(f_input)
        
        f_grad_input = apply_format(grad_input, mode)
        f_grad_weight = apply_format(grad_weight, mode)
        f_grad_bias = apply_format(f_grad_output.sum(dim=0), mode) if f_bias is not None else None

        return f_grad_input, f_grad_weight, f_grad_bias, None

# ==========================================
# 3. HARDWARE LAYERS
# ==========================================
class HardwareConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding=1, mode='FP32'):
        super().__init__()
        self.mode = mode
        self.stride = 1
        self.padding = padding
        self.weight = nn.Parameter(torch.empty(out_channels, in_channels, kernel_size, kernel_size))
        self.bias = nn.Parameter(torch.zeros(out_channels))
        nn.init.kaiming_normal_(self.weight, mode='fan_out', nonlinearity='relu')
        self.weight.data = apply_format(self.weight.data, mode)

    def forward(self, x):
        return PureFormatConv2dFunc.apply(x, self.weight, self.bias, self.stride, self.padding, self.mode)

class HardwareLinear(nn.Module):
    def __init__(self, in_features, out_features, mode='FP32'):
        super().__init__()
        self.mode = mode
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias = nn.Parameter(torch.zeros(out_features))
        nn.init.kaiming_normal_(self.weight, mode='fan_out', nonlinearity='relu')
        self.weight.data = apply_format(self.weight.data, mode)

    def forward(self, x):
        return PureFormatLinearFunc.apply(x, self.weight, self.bias, self.mode)

# ==========================================
# 4. VGG-16 ARCHITECTURE
# ==========================================
class VGG16_Hardware(nn.Module):
    def __init__(self, mode, num_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            HardwareConv2d(3, 64, 3, padding=1, mode=mode), nn.ReLU(),
            HardwareConv2d(64, 64, 3, padding=1, mode=mode), nn.ReLU(), nn.MaxPool2d(2, 2),
            HardwareConv2d(64, 128, 3, padding=1, mode=mode), nn.ReLU(),
            HardwareConv2d(128, 128, 3, padding=1, mode=mode), nn.ReLU(), nn.MaxPool2d(2, 2),
            HardwareConv2d(128, 256, 3, padding=1, mode=mode), nn.ReLU(),
            HardwareConv2d(256, 256, 3, padding=1, mode=mode), nn.ReLU(),
            HardwareConv2d(256, 256, 3, padding=1, mode=mode), nn.ReLU(), nn.MaxPool2d(2, 2),
            HardwareConv2d(256, 512, 3, padding=1, mode=mode), nn.ReLU(),
            HardwareConv2d(512, 512, 3, padding=1, mode=mode), nn.ReLU(),
            HardwareConv2d(512, 512, 3, padding=1, mode=mode), nn.ReLU(), nn.MaxPool2d(2, 2),
            HardwareConv2d(512, 512, 3, padding=1, mode=mode), nn.ReLU(),
            HardwareConv2d(512, 512, 3, padding=1, mode=mode), nn.ReLU(),
            HardwareConv2d(512, 512, 3, padding=1, mode=mode), nn.ReLU(), nn.MaxPool2d(2, 2)
        )
        self.classifier = nn.Sequential(
            HardwareLinear(512 * 1 * 1, 4096, mode=mode), nn.ReLU(), nn.Dropout(),
            HardwareLinear(4096, 4096, mode=mode), nn.ReLU(), nn.Dropout(),
            HardwareLinear(4096, num_classes, mode=mode)
        )

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)

# ==========================================
# 5. MASTER INFERENCE LOOP
# ==========================================
if __name__ == "__main__":
    MODES = ['FP32', 'FP16', 'FXP32', 'FXP16', 'POSIT32_2', 'POSIT16_1', 'POSIT8_0']
    
    transform = transforms.Compose([
        transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    # Exactly 10,000 images strictly held out for final testing
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=False, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=False)

    for mode in MODES:
        print(f"\n{'='*50}")
        print(f"EVALUATING NATIVE FORMAT: {mode}")
        print(f"{'='*50}")
        
        weight_file = f"vgg16_native_weights_{mode}.pth"
        if not os.path.exists(weight_file):
            print(f"SKIPPING: Weights file {weight_file} not found. Ensure training finished.")
            continue

        if 'POSIT' in mode:
            device = torch.device('cpu')
        else:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        model = VGG16_Hardware(mode=mode).to(device)
        model.load_state_dict(torch.load(weight_file, map_location=device), strict=True)
        model.eval()

        top1_correct = 0
        top5_correct = 0
        total = 0

        # Disable gradient calculations for pure inference speed
        with torch.no_grad():
            for i, (inputs, labels) in enumerate(testloader):
                inputs, labels = inputs.to(device), labels.to(device)
                
                # Format the raw image pixels before they hit the first layer
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
