import os
import torch
import torch.nn as nn
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from parallel_experts import RandomMoE, TaskMoE


class TransformerMoETaskGating(nn.Module):
    def __init__(self, task_dict, embed_dim=128, depth=4, num_heads=6, mlp_ratio=4., qkv_bias=True, 
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.,
                 num_attn_experts=8, head_dim=None, att_w_topk_loss=0.0, att_limit_k=0, 
                 num_ffd_experts=8, ffd_heads=2, ffd_noise=True,
                 moe_type='normal',
                 switchloss=0.01 * 1, zloss=0.001 * 1, w_topk_loss= 0.0, limit_k=0, 
                 w_MI = 0.,
                 noisy_gating=True,
                 post_layer_norm=False,
                 twice_mlp=False,
                 twice_attn=False,
                 return_hidden = False,
                 **kwargs):
        super(TransformerMoETaskGating, self).__init__()


        self.return_hidden = return_hidden
        self.task_dict = task_dict
        
        self.moe_type = moe_type
        self.depth = depth

        self.w_topk_loss = w_topk_loss
        self.ismoe = True
        self.task_num = len(self.task_dict)
        self.fc_norm = nn.LayerNorm(embed_dim)
        self.drop = nn.Dropout(0.3)

        self.task_heads = []
        
        for t, num_t in self.task_dict.items():
            img_type = t
            self.task_heads.append(
                    nn.Sequential(
                        nn.LayerNorm(embed_dim),
                        nn.Linear(embed_dim, num_t)
                    )
                )

        self.task_heads = nn.ModuleList(self.task_heads)
        self.task_embedding = nn.Parameter(torch.randn(1, len(self.task_dict), embed_dim))       
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        
        self.twice_mlp = twice_mlp

        self.blocks = nn.Sequential(*[
            MoEnhanceTaskBlock( dim=embed_dim, num_heads=num_heads,
                task_num=self.task_num,
                num_attn_experts=num_attn_experts, head_dim=head_dim,
                use_moe_mlp=(twice_mlp==False or (i%2)==1),
                use_moe_attn=(twice_attn==False or (i%2)==0),
                )
            for i in range(depth)])


    def moa_init_weight(self, module):
        if isinstance(module, (nn.Linear)):
            module.weight.data.fill_(0.00)

    def get_zloss(self):
        z_loss = 0
        for blk in self.blocks:
            if hasattr(blk.attn, 'num_experts'):
                aux_loss = blk.attn.q_proj.get_aux_loss_and_clear()
                z_loss = z_loss + aux_loss

            if hasattr(blk.mlp, 'num_experts'):
                aux_loss = blk.mlp.get_aux_loss_and_clear()
                z_loss = z_loss + aux_loss
        return z_loss

    def get_topkloss(self):
        z_loss = 0
        for blk in self.blocks:
            if hasattr(blk.attn, 'num_experts'):
                aux_loss = blk.attn.q_proj.get_topk_loss_and_clear()
                z_loss = z_loss + aux_loss

            # break
            if hasattr(blk.mlp, 'num_experts'):
                aux_loss = blk.mlp.get_topk_loss_and_clear()
                z_loss = z_loss + aux_loss
        return z_loss

    def all_clear(self):
        for blk in self.blocks:
            aux_loss = blk.attn.q_proj.init_aux_statistics()
            if hasattr(blk.mlp, 'num_experts'):
                aux_loss = blk.mlp.init_aux_statistics()

    def visualize(self, vis_head=False, vis_mlp=False, model_name=''):
        all_list = []
        torch.set_printoptions(precision=4, sci_mode=False)

        for depth, blk in enumerate(self.blocks):
            layer_list = {}
            for i, the_type in enumerate(self.task_dict.keys()):

                layer_list[the_type] = []
                if hasattr(blk.attn, 'num_experts'):
                    if blk.attn.num_experts > blk.attn.num_heads:
                        # _sum = blk.attn.q_proj.task_gate_freq[i].sum()
                        # layer_list[the_type].append((blk.attn.q_proj.task_gate_freq[i] / _sum * 100).tolist())
                        layer_list[the_type].append(blk.attn.q_proj.task_gate_freq[i])
                if hasattr(blk.mlp, 'num_experts'):
                    if blk.mlp.num_experts > blk.mlp.k:
                        # _sum = blk.mlp.task_gate_freq[i].sum()
                        # layer_list[the_type].append((blk.mlp.task_gate_freq[i] / _sum * 100).tolist())
                        layer_list[the_type].append(blk.mlp.task_gate_freq[i])
            all_list.append(layer_list)
        print(all_list)

        torch.save(all_list, str(model_name) + '_vis.t7')
            
    

    def forward_features(self, x):
        
        B = x.shape[0]
        
#         pos_embed = 
        
#         x = x + pos_embed

        x_before = self.drop(x)

        # apply Transformer blocks

        output = {}
        z_loss = 0
        for t, the_type in enumerate(self.task_dict.keys()):
            x = x_before
            for blk in self.blocks:
                x = x + self.task_embedding[:, t:t+1, :]
                x, _ = blk(x, t)

            x = x[:, 1:, :].mean(dim=1)
            x = self.fc_norm(x)

            output[the_type] = x
            if self.w_topk_loss > 0.0:
                z_loss = z_loss + self.get_topkloss()
        
        return output, z_loss

    def forward(self, x, get_flop=False, get_z_loss=False):

        output, z_loss = self.forward_features(x)
        if self.return_hidden:
            hidden = output
        for t, the_type in enumerate(self.task_dict.keys()):
            output[the_type] = self.task_heads[t](output[the_type])

        if get_flop:
            return output['class_object']

        if self.return_hidden:
            return hidden, output, z_loss + self.get_zloss()
        else:
            return output, z_loss + self.get_zloss()

class MoEnhanceTaskBlock(nn.Module):

    def __init__(self, dim, num_heads, num_attn_experts=24, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 num_ffd_experts=16, ffd_heads=2, ffd_noise=True,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, head_dim=None, init_values=None, z_weight=0.000,
                 post_layer_norm=False,
                 task_num=9,
                 noisy_gating=True,
                 att_w_topk_loss=0.0, att_limit_k=0, 
                 cvloss=0, switchloss=0.01 * 1, zloss=0.001 * 1, w_topk_loss=0.0, limit_k=0, 
                 w_MI = 0.,
                 use_moe_mlp=True,
                 use_moe_attn=True,
                 sample_topk=0, moe_type='normal'):
        super().__init__()
        self.task_num = task_num
        self.norm1 = norm_layer(dim)
        self.use_moe_attn = use_moe_attn
        self.attn = MoETaskAttention(
            dim, task_num=task_num, noisy_gating=noisy_gating, num_heads=num_heads, num_experts=num_attn_experts, head_dim=head_dim, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop,
            cvloss=cvloss, switchloss=switchloss, zloss=zloss, w_MI=w_MI, w_topk_loss=att_w_topk_loss, limit_k=att_limit_k, sample_topk=sample_topk, moe_type=moe_type)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)

        self.use_moe_mlp = use_moe_mlp

        self.mlp = TaskMoE(dim,
                mlp_hidden_dim // ffd_heads, num_ffd_experts, ffd_heads,
                bias=True,
                acc_aux_loss=True, 
                cvloss=cvloss,
                switchloss=switchloss,
                zloss=zloss,
                w_topk_loss=w_topk_loss,
                w_MI=w_MI,
                limit_k=limit_k,
                task_num=task_num,
                activation=nn.Sequential(
                    nn.GELU(),
                ),
                noisy_gating=ffd_noise
            )
        assert z_weight == 0

    def forward(self, x, task_bh, mask=None):

        y, z_loss = self.attn(self.norm1(x), task_bh, mask=mask)
        x = x + self.drop_path(y)

        y, aux_loss = self.mlp(self.norm2(x), task_bh)
        x = x + self.drop_path(y)

        return x, z_loss + aux_loss

    
    
class MoETaskAttention(nn.Module):
    def __init__(self, dim, noisy_gating=True, task_num=9, num_experts=24, num_heads=8, head_dim=None, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.,
        sample_topk=2, cvloss=0, switchloss=0.01 * 10, zloss=0.001 * 1, w_topk_loss=0.1, w_MI=0., limit_k=0, moe_type='normal'):
        super().__init__()
        self.task_num = task_num
        self.num_experts = num_experts
        self.sample_topk = sample_topk

        self.num_heads = num_heads
        if head_dim is None:
            head_dim = dim // num_heads
        self.head_dim = head_dim
        inner_dim = num_heads * head_dim
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5
        self.moe_type = moe_type

        self.q_proj = TaskMoE(dim, head_dim, num_experts, num_heads, noisy_gating=noisy_gating, w_MI=w_MI, acc_aux_loss=True, task_num=task_num, cvloss=cvloss, switchloss=switchloss, zloss=zloss, w_topk_loss=w_topk_loss, limit_k=limit_k)

        self.kv_proj = nn.Sequential(
            nn.Linear(dim, head_dim * 2),
        )

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, task_bh, mask=None):

        B, N, C = x.shape        
        q, aux_loss = self.q_proj.map(x, task_bh, sample_topk=self.sample_topk)
        k, v = self.kv_proj(x).chunk(2, dim=-1)

        q = q.reshape(B, N, self.num_heads, self.head_dim)
        k = k.reshape(B, N, self.head_dim)
        v = v.reshape(B, N, self.head_dim)

        attn = torch.einsum('bihd,bjd->bhij', q, k) * self.scale
        # attn = attn.premute(0,3,1,2) # b, h, i, j

        if mask is not None:
            mask = mask.bool()
            attn = attn.masked_fill(~mask[:, None, None, :], float("-inf"))

        # For rare cases, the attention weights are inf due to the mix-precision training.
        # We clamp the tensor to the max values of the current data type
        # This is different from MAE training as we don't observe such cases on image-only MAE.
        if torch.isinf(attn).any():
            clamp_value = torch.finfo(attn.dtype).max-1000
            attn = torch.clamp(attn, min=-clamp_value, max=clamp_value)

        attn = attn.softmax(dim=-1)

        attn = self.attn_drop(attn)

        attn = torch.einsum('bhij,bjd->bihd', attn, v)

        if self.moe_type == 'FLOP':
            x = self.q_proj.dispatch(
                    attn.reshape(B, N, self.num_heads, self.head_dim).contiguous(), 
                    self.out_proj
                )
        else:
            x = self.q_proj.reduce(attn)
        x = self.proj_drop(x)
        return x, aux_loss
