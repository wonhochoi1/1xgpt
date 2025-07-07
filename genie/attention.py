import torch
from torch import nn
import math
## UNCOMMENT IF XFORMERS_DISABLED == false
# from xformers.ops import LowerTriangularMask, memory_efficient_attention, unbind
import os


#allows for using xformers or not - xformers are not supported on MacOS 
XFORMERS_DISABLED = os.environ.get("XFORMERS_DISABLED", "false").lower() == "true"

### IMPLEMENTATION FOR WINDOWED AND AXIAL ATTENTION ###
class WindowedAttention(nn.Module):
    """Windowed attention for efficient processing of long sequences"""
    def __init__(
        self,
        num_heads: int,
        d_model: int,
        window_size: int = 64,
        qkv_bias: bool = False,
        proj_bias: bool = True,
        qk_norm: bool = True,
        use_mup: bool = True,
        attn_drop: float = 0.0,
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.window_size = window_size
        
        # Scaling by 8 to be equal when head_dim=64
        self.scale = 8/self.head_dim if use_mup else self.head_dim**-0.5
        self.qkv = nn.Linear(d_model, d_model * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(d_model, d_model, bias=proj_bias)
        self.qk_norm = qk_norm
        if self.qk_norm:
            self.norm = nn.LayerNorm(self.head_dim, eps=1e-05)

    def forward(self, x: torch.Tensor, causal: bool = False) -> torch.Tensor:
        B, N, C = x.shape
        
        # If sequence length is shorter than window size, use standard attention
        if N <= self.window_size:
            return self._standard_attention(x, causal)
        
        # Apply windowed attention
        return self._windowed_attention(x, causal)
    
    def _standard_attention(self, x: torch.Tensor, causal: bool = False) -> torch.Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        if self.qk_norm:
            q = self.norm(q)
            k = self.norm(k)
            q = q.to(dtype=v.dtype)
            k = k.to(dtype=v.dtype)
        
        q *= self.scale
        attn = q @ k.transpose(-2, -1)

        if causal:
            mask_value = -torch.finfo(attn.dtype).max
            i, j = attn.shape[-2:]            
            mask = ~torch.tril(torch.ones(i, j)).bool().to(attn.device)
            attn = attn.masked_fill(mask, mask_value)

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        return x
    
    def _windowed_attention(self, x: torch.Tensor, causal: bool = False) -> torch.Tensor:
        B, N, C = x.shape
        num_windows = (N + self.window_size - 1) // self.window_size
        
        # Pad sequence to be divisible by window size
        pad_length = num_windows * self.window_size - N
        if pad_length > 0:
            x = torch.cat([x, torch.zeros(B, pad_length, C, device=x.device, dtype=x.dtype)], dim=1)
            N = x.shape[1]
        
        # Reshape to windows
        x_windows = x.view(B, num_windows, self.window_size, C)
        
        # Apply attention within each window
        outputs = []
        for i in range(num_windows):
            window_output = self._standard_attention(x_windows[:, i], causal)
            outputs.append(window_output)
        
        # Concatenate window outputs
        x = torch.cat(outputs, dim=1)
        
        # Remove padding if added
        if pad_length > 0:
            x = x[:, :N-pad_length]
        
        return x


class AxialAttention(nn.Module):
    """Axial attention for 2D spatial data"""
    def __init__(
        self,
        num_heads: int,
        d_model: int,
        height: int = 16,
        width: int = 16,
        qkv_bias: bool = False,
        proj_bias: bool = True,
        qk_norm: bool = True,
        use_mup: bool = True,
        attn_drop: float = 0.0,
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.height = height
        self.width = width
        
        # Scaling by 8 to be equal when head_dim=64
        self.scale = 8/self.head_dim if use_mup else self.head_dim**-0.5
        
        # Separate QKV projections for row and column attention
        self.qkv_row = nn.Linear(d_model, d_model * 3, bias=qkv_bias)
        self.qkv_col = nn.Linear(d_model, d_model * 3, bias=qkv_bias)
        
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(d_model, d_model, bias=proj_bias)
        self.qk_norm = qk_norm
        if self.qk_norm:
            self.norm_row = nn.LayerNorm(self.head_dim, eps=1e-05)
            self.norm_col = nn.LayerNorm(self.head_dim, eps=1e-05)

    def forward(self, x: torch.Tensor, causal: bool = False) -> torch.Tensor:
        B, N, C = x.shape
        
        # Reshape to 2D spatial format
        if N == self.height * self.width:
            x_2d = x.reshape(B, self.height, self.width, C)
        else:
            # If not spatial data, fall back to standard attention
            return self._standard_attention(x, causal)
        
        # Apply row-wise attention
        x_row = self._row_attention(x_2d, causal)
        
        # Apply column-wise attention
        x_col = self._column_attention(x_row, causal)
        
        # Reshape back to sequence format
        x = x_col.reshape(B, N, C)
        return x
    
    def _row_attention(self, x: torch.Tensor, causal: bool = False) -> torch.Tensor:
        B, H, W, C = x.shape
        
        # Process each row independently
        outputs = []
        for h in range(H):
            row = x[:, h, :, :]  # (B, W, C)
            row_output = self._attention_1d(row, self.qkv_row, self.norm_row if self.qk_norm else None, causal)
            outputs.append(row_output)
        
        return torch.stack(outputs, dim=1)  # (B, H, W, C)
    
    def _column_attention(self, x: torch.Tensor, causal: bool = False) -> torch.Tensor:
        B, H, W, C = x.shape
        
        # Process each column independently
        outputs = []
        for w in range(W):
            col = x[:, :, w, :]  # (B, H, C)
            col_output = self._attention_1d(col, self.qkv_col, self.norm_col if self.qk_norm else None, causal)
            outputs.append(col_output)
        
        # Transpose back to original shape
        return torch.stack(outputs, dim=2).transpose(1, 2)  # (B, H, W, C)
    
    def _attention_1d(self, x: torch.Tensor, qkv_layer: nn.Linear, norm_layer: nn.LayerNorm, causal: bool = False) -> torch.Tensor:
        B, N, C = x.shape
        qkv = qkv_layer(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        if norm_layer is not None:
            q = norm_layer(q)
            k = norm_layer(k)
            q = q.to(dtype=v.dtype)
            k = k.to(dtype=v.dtype)
        
        q *= self.scale
        attn = q @ k.transpose(-2, -1)

        if causal:
            mask_value = -torch.finfo(attn.dtype).max
            i, j = attn.shape[-2:]            
            mask = ~torch.tril(torch.ones(i, j)).bool().to(attn.device)
            attn = attn.masked_fill(mask, mask_value)

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        return x
    
    def _standard_attention(self, x: torch.Tensor, causal: bool = False) -> torch.Tensor:
        B, N, C = x.shape
        qkv = self.qkv_row(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        if self.qk_norm:
            q = self.norm_row(q)
            k = self.norm_row(k)
            q = q.to(dtype=v.dtype)
            k = k.to(dtype=v.dtype)
        
        q *= self.scale
        attn = q @ k.transpose(-2, -1)

        if causal:
            mask_value = -torch.finfo(attn.dtype).max
            i, j = attn.shape[-2:]            
            mask = ~torch.tril(torch.ones(i, j)).bool().to(attn.device)
            attn = attn.masked_fill(mask, mask_value)

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        return x


###

class BasicSelfAttention(nn.Module):
    def __init__(
        self,
        num_heads: int,
        d_model: int,
        qkv_bias: bool = False,
        proj_bias: bool = True,
        qk_norm: bool = True,
        use_mup: bool = True,
        attn_drop: float = 0.0,
    ) -> None:
        super().__init__()

        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        # Scaling by 8 to be equal when head_dim=64
        self.scale = 8/self.head_dim if use_mup else self.head_dim**-0.5
        self.qkv = nn.Linear(d_model, d_model * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(d_model, d_model, bias=proj_bias)
        self.qk_norm = qk_norm
        if self.qk_norm:
            # qk normalization https://arxiv.org/pdf/2302.05442
            # Note that LN is done in fp32, so they have to be
            self.norm = nn.LayerNorm(self.head_dim, eps=1e-05)

    def forward(self, x: torch.Tensor, causal: bool = False) -> torch.Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)

        q, k, v = qkv[0], qkv[1], qkv[2]

        if self.qk_norm:
            q = self.norm(q)
            k = self.norm(k)
            # LN done in float32, cast back to bf16
            q = q.to(dtype=v.dtype)
            k = k.to(dtype=v.dtype)
        q *= self.scale
        attn = q @ k.transpose(-2, -1)

        if causal:
            mask_value = -torch.finfo(attn.dtype).max
            i, j = attn.shape[-2:]            
            mask = ~torch.tril(torch.ones(i, j)).bool().to(attn.device)
            attn = attn.masked_fill(mask, mask_value)

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        return x


class MemoryEfficientAttention(BasicSelfAttention):
    # NOTE: Mem-eff attention from xformers is actually Flash Attention 2
        
    def forward(self, x: torch.Tensor, causal: bool = False) -> torch.Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        q, k, v = unbind(qkv, 2)
        if self.qk_norm:
            q = self.norm(q)
            k = self.norm(k)
            # LN done in float32, cast back to bf16
            q = q.to(dtype=v.dtype)
            k = k.to(dtype=v.dtype)

        attn_bias = LowerTriangularMask() if causal else None
        x = memory_efficient_attention(q, k, v, attn_bias=attn_bias, scale=self.scale)
        x = x.reshape([B, N, C])

        x = self.proj(x)
        return x

# Use efficient attention by default for better memory efficiency with 4K tokens
if XFORMERS_DISABLED:
    # Use windowed attention as the default efficient attention
    SelfAttention = WindowedAttention
else:
    SelfAttention = MemoryEfficientAttention