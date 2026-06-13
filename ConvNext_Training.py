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
# 1. NATIVE FORMAT CASTER (WITH MULTIPROCESSING)
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
# 2. PURE FORMAT AUTOGRAD (FORWARD & BACKWARD)
# ==========================================
class PureFormatConv2dFunc(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input_t, weight_t, bias_t, stride, padding, groups, mode):
        f_input = apply_format(input_t, mode)
        f_weight = apply_format(weight_t, mode)
        f_bias = apply_format(bias_t, mode) if bias_t is not None else None

        out = F.conv2d(f_input, f_weight, f_bias, stride=stride, padding=padding, groups=groups)
        f_out = apply_format(out, mode)

        ctx.save_for_backward(f_input, f_weight, f_bias)
        ctx.stride = stride
        ctx.padding = padding
        ctx.groups = groups
        ctx.mode = mode
        return f_out

    @staticmethod
    def backward(ctx, grad_output):
        f_input, f_weight, f_bias = ctx.saved_tensors
        mode = ctx.mode

        f_grad_output = apply_format(grad_output, mode)

        grad_input = torch.nn.grad.conv2d_input(f_input.shape, f_weight, f_grad_output, stride=ctx.stride, padding=ctx.padding, groups=ctx.groups)
        grad_weight = torch.nn.grad.conv2d_weight(f_input, f_weight.shape, f_grad_output, stride=ctx.stride, padding=ctx.padding, groups=ctx.groups)
        
        f_grad_input = apply_format(grad_input, mode)
        f_grad_weight = apply_format(grad_weight, mode)
        f_grad_bias = apply_format(f_grad_output.sum(dim=(0, 2, 3)), mode) if f_bias is not None else None

        return f_grad_input, f_grad_weight, f_grad_bias, None, None, None, None

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
# 3. HARDWARE LAYERS (UPGRADED FOR CONVNEXT)
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
        nn.init.kaiming_normal_(self.weight, mode='fan_out', nonlinearity='relu')
        self.weight.data = apply_format(self.weight.data, mode)
        self.bias.data = apply_format(self.bias.data, mode)

    def forward(self, x):
        return PureFormatConv2dFunc.apply(x, self.weight, self.bias, self.stride, self.padding, self.groups, self.mode)

class HardwareLinear(nn.Module):
    def __init__(self, in_features, out_features, mode='FP32'):
        super().__init__()
        self.mode = mode
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias = nn.Parameter(torch.zeros(out_features))
        nn.init.kaiming_normal_(self.weight, mode='fan_out', nonlinearity='relu')
        self.weight.data = apply_format(self.weight.data, mode)
        self.bias.data = apply_format(self.bias.data, mode)

    def forward(self, x):
        return PureFormatLinearFunc.apply(x, self.weight, self.bias, self.mode)

class HardwareLayerNormChannelsFirst(nn.Module):
    def __init__(self, normalized_shape, eps=1e-6, mode='FP32'):
        super().__init__()
        self.mode = mode
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.weight.data = apply_format(self.weight.data, mode)
        self.bias.data = apply_format(self.bias.data, mode)

    def forward(self, x):
        f_x = apply_format(x, self.mode)
        u = f_x.mean(1, keepdim=True)
        s = (f_x - u).pow(2).mean(1, keepdim=True)
        normed = (f_x - u) / torch.sqrt(s + self.eps)
        
        f_weight = apply_format(self.weight[:, None, None], self.mode)
        f_bias = apply_format(self.bias[:, None, None], self.mode)
        out = f_weight * normed + f_bias
        return apply_format(out, self.mode)

# ==========================================
# 4. CUSTOM PURE FORMAT OPTIMIZER
# ==========================================
class PureFormatSGD(torch.optim.Optimizer):
    def __init__(self, params, lr, mode):
        defaults = dict(lr=lr, mode=mode)
        super(PureFormatSGD, self).__init__(params, defaults)

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            mode = group['mode']
            lr_t = torch.tensor(group['lr'], device=self.param_groups[0]['params'][0].device)
            f_lr = apply_format(lr_t, mode)

            for p in group['params']:
                if p.grad is not None:
                    f_grad = apply_format(p.grad, mode)
                    update = apply_format(f_lr * f_grad, mode)
                    p.data = apply_format(p.data - update, mode)

# ==========================================
# 5. CONVNEXT ARCHITECTURE
# ==========================================
class HardwareConvNeXtBlock(nn.Module):
    def __init__(self, dim, mode='FP32'):
        super().__init__()
        self.mode = mode
        self.dwconv = HardwareConv2d(dim, dim, kernel_size=7, stride=1, padding=3, groups=dim, mode=mode)
        
        # LayerNorm applied manually on last dim to maintain format strictness
        self.norm_weight = nn.Parameter(torch.ones(dim))
        self.norm_bias = nn.Parameter(torch.zeros(dim))
        
        self.pwconv1 = HardwareLinear(dim, 4 * dim, mode=mode)
        self.act = nn.GELU()
        self.pwconv2 = HardwareLinear(4 * dim, dim, mode=mode)
        self.gamma = nn.Parameter(1e-6 * torch.ones((dim)))
        
        self.norm_weight.data = apply_format(self.norm_weight.data, mode)
        self.norm_bias.data = apply_format(self.norm_bias.data, mode)
        self.gamma.data = apply_format(self.gamma.data, mode)

    def forward(self, x):
        input_tensor = apply_format(x, self.mode)
        
        x = self.dwconv(input_tensor)
        x = x.permute(0, 2, 3, 1) # (N, C, H, W) -> (N, H, W, C)
        
        # Pure Format LayerNorm
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
        
        x = x.permute(0, 3, 1, 2) # (N, H, W, C) -> (N, C, H, W)
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
            downsample_layer = nn.Sequential(
                HardwareLayerNormChannelsFirst(dims[i], mode=mode),
                HardwareConv2d(dims[i], dims[i+1], kernel_size=2, stride=2, padding=0, mode=mode),
            )
            self.downsample_layers.append(downsample_layer)

        self.stages = nn.ModuleList() 
        for i in range(4):
            stage = nn.Sequential(*[HardwareConvNeXtBlock(dim=dims[i], mode=mode) for _ in range(depths[i])])
            self.stages.append(stage)
            
        self.norm_weight = nn.Parameter(torch.ones(dims[-1]))
        self.norm_bias = nn.Parameter(torch.zeros(dims[-1]))
        self.norm_weight.data = apply_format(self.norm_weight.data, mode)
        self.norm_bias.data = apply_format(self.norm_bias.data, mode)
        
        self.head = HardwareLinear(dims[-1], num_classes, mode=mode)

    def forward(self, x):
        x = apply_format(x, self.mode)
        for i in range(4):
            x = self.downsample_layers[i](x)
            x = self.stages[i](x)
            
        x = x.mean([-2, -1]) # Global Average Pooling
        
        f_x = apply_format(x, self.mode)
        u = f_x.mean(-1, keepdim=True)
        s = (f_x - u).pow(2).mean(-1, keepdim=True)
        normed = (f_x - u) / torch.sqrt(s + 1e-6)
        
        f_nw = apply_format(self.norm_weight, self.mode)
        f_nb = apply_format(self.norm_bias, self.mode)
        x = apply_format(f_nw * normed + f_nb, self.mode)
        
        return self.head(x)

# ==========================================
# 6. EXECUTION LOOP
# ==========================================
if __name__ == "__main__":
    MODES = ['FP32', 'FP16', 'FXP32', 'FXP16', 'POSIT32_2', 'POSIT16_1', 'POSIT8_0']
    EPOCHS = 20
    
    LR_MAP = {
        'FP32': 0.001,
        'FP16': 0.001,
        'FXP32': 0.005,
        'FXP16': 0.1,
        'POSIT32_2': 0.001,
        'POSIT16_1': 0.05,
        'POSIT8_0': 0.2
    }
    
    transform = transforms.Compose([
        transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    full_trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    
    train_subset = torch.utils.data.Subset(full_trainset, range(0, 40000))
    val_subset = torch.utils.data.Subset(full_trainset, range(40000, 50000))
    
    trainloader = torch.utils.data.DataLoader(train_subset, batch_size=64, shuffle=True)
    valloader = torch.utils.data.DataLoader(val_subset, batch_size=64, shuffle=False)

    for mode in MODES:
        print(f"\n{'='*50}")
        print(f"STARTING CONVNEXT DATAPATH TRAINING: {mode}")
        print(f"{'='*50}")
        
        if 'POSIT' in mode:
            device = torch.device('cpu')
        else:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        model = ConvNeXt_Hardware(mode=mode).to(device)
        criterion = nn.CrossEntropyLoss()
        
        current_lr = LR_MAP[mode]
        print(f"Initialized PureFormatSGD with Learning Rate: {current_lr}")
        optimizer = PureFormatSGD(model.parameters(), lr=current_lr, mode=mode)

        for epoch in range(EPOCHS):
            model.train()
            running_loss = 0.0
            for i, (inputs, labels) in enumerate(trainloader):
                inputs, labels = inputs.to(device), labels.to(device)
                
                optimizer.zero_grad()
                inputs = apply_format(inputs, mode)
                
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
                
            avg_train_loss = running_loss / len(trainloader)

            model.eval()
            val_correct = 0
            val_total = 0
            with torch.no_grad():
                for val_inputs, val_labels in valloader:
                    val_inputs, val_labels = val_inputs.to(device), val_labels.to(device)
                    val_inputs = apply_format(val_inputs, mode)
                    val_outputs = model(val_inputs)
                    _, val_predicted = torch.max(val_outputs.data, 1)
                    val_total += val_labels.size(0)
                    val_correct += (val_predicted == val_labels).sum().item()

            val_accuracy = 100 * val_correct / val_total
            print(f"[{mode}] Epoch [{epoch+1}/{EPOCHS}] | Train Loss: {avg_train_loss:.3f} | Val Acc: {val_accuracy:.2f}%")

        save_path = f"convnext_native_weights_{mode}.pth"
        torch.save(model.state_dict(), save_path)
        
        hw_file_path = f"convnext_native_hardware_{mode}.txt"
        with open(hw_file_path, "w") as f:
            for name, param in model.named_parameters():
                if 'weight' in name or 'bias' in name:
                    f.write(f"// ==== LAYER: {name} | MODE: {mode} ====\n")
                    raw_weights = param.data.flatten().cpu().numpy()
                    for val in raw_weights:
                        f.write(f"{val}\n")
