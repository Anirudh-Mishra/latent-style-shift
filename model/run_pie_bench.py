from diffusers import LCMScheduler
from pipeline_ead import EditPipeline
import os
import gradio as gr
import torch
from PIL import Image, ImageFilter
import torch.nn.functional as nnf
from typing import Optional, Union, Tuple, List, Callable, Dict
import abc
import ptp_utils
import utils
import numpy as np
import seq_aligner
import math
import argparse
import json

LOW_RESOURCE = False
MAX_NUM_WORDS = 77

is_colab = utils.is_google_colab()
colab_instruction = "" if is_colab else """
Colab Instuction"""

torch_dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
model_id_or_path = "SimianLuo/LCM_Dreamshaper_v7"
device_print = "GPU 🔥" if torch.cuda.is_available() else "CPU 🥶"
device = "cuda" if torch.cuda.is_available() else "cpu"

if is_colab:
    scheduler = LCMScheduler.from_config(model_id_or_path, subfolder="scheduler")
    pipe = EditPipeline.from_pretrained(model_id_or_path, scheduler=scheduler, torch_dtype=torch_dtype)
else:
    scheduler = LCMScheduler.from_config(model_id_or_path, use_auth_token=os.environ.get("USER_TOKEN"), subfolder="scheduler")
    pipe = EditPipeline.from_pretrained(model_id_or_path, use_auth_token=os.environ.get("USER_TOKEN"), scheduler=scheduler, torch_dtype=torch_dtype)

# Runtime backbone is selected in main() from CLI args.
_backbone = "unet"

tokenizer = pipe.tokenizer
encoder = pipe.text_encoder

if torch.cuda.is_available():
    pipe = pipe.to("cuda")

# Disable safety checker for research/benchmark use — out-of-range latents from
# an undertrained model trigger false positives and return black images.
pipe.safety_checker = None


class LocalBlend:
    
    def get_mask(self,x_t,maps,word_idx, thresh, i):
        maps = maps * word_idx.reshape(1,1,1,1,-1)
        maps = (maps[:,:,:,:,1:self.len-1]).mean(0,keepdim=True)
        maps = (maps).max(-1)[0]
        maps = nnf.interpolate(maps, size=(x_t.shape[2:]))
        maps = maps / maps.max(2, keepdim=True)[0].max(3, keepdim=True)[0]
        mask = maps > thresh
        return mask


    def save_image(self,mask,i, caption):
        image = mask[0, 0, :, :]
        image = 255 * image / image.max()
        # print(image.shape)
        image = image.unsqueeze(-1).expand(*image.shape, 3)
        # print(image.shape)
        image = image.cpu().numpy().astype(np.uint8)
        image = np.array(Image.fromarray(image).resize((256, 256)))
        if not os.path.exists(f"inter/{caption}"):
           os.mkdir(f"inter/{caption}") 
        ptp_utils.save_images(image, f"inter/{caption}/{i}.jpg")
        

    def __call__(self, i, x_s, x_t, x_m, attention_store, alpha_prod, temperature=0.15, use_xm=False):
        is_uvit = _backbone == "uvit"

        if is_uvit:
            down_maps = attention_store.get("down_cross", [])
            up_maps = attention_store.get("up_cross", [])
            n_down = len(down_maps)
            n_up = len(up_maps)
            maps = down_maps[max(0, n_down-2):] + up_maps[:min(3, n_up)]
            if len(maps) == 0:
                return x_m, x_t
            h, w = x_t.shape[2], x_t.shape[3]
            # Read patch_size from the backbone rather than hardcoding
            patch_size = pipe.unet.backbone.patch_size if hasattr(pipe.unet, 'backbone') else 2
            h_p, w_p = h // patch_size, w // patch_size
            maps = [item[:, 1:h_p*w_p+1, :].reshape(2, -1, 1, h_p, w_p, MAX_NUM_WORDS) for item in maps]
        else:
            maps = attention_store["down_cross"][2:4] + attention_store["up_cross"][:3]
            h, w = x_t.shape[2], x_t.shape[3]
            h, w = ((h+1)//2+1)//2, ((w+1)//2+1)//2
            maps = [item.reshape(2, -1, 1, h // int((h*w/item.shape[-2])**0.5),  w // int((h*w/item.shape[-2])**0.5), MAX_NUM_WORDS) for item in maps]
        maps = torch.cat(maps, dim=1)
        maps_s = maps[0,:]
        maps_m = maps[1,:]
        thresh_e = temperature / alpha_prod ** (0.5)
        if thresh_e < self.thresh_e:
          thresh_e = self.thresh_e
        thresh_m = self.thresh_m
        mask_e = self.get_mask(x_t, maps_m, self.alpha_e, thresh_e, i)
        mask_m = self.get_mask(x_t, maps_s, (self.alpha_m-self.alpha_me), thresh_m, i)
        mask_me = self.get_mask(x_t, maps_m, self.alpha_me, self.thresh_e, i)
        if self.save_inter:
            self.save_image(mask_e,i,"mask_e")
            self.save_image(mask_m,i,"mask_m")
            self.save_image(mask_me,i,"mask_me")

        if self.alpha_e.sum() == 0:
          x_t_out = x_t
        else:
          x_t_out = torch.where(mask_e, x_t, x_m)
        x_t_out = torch.where(mask_m, x_s, x_t_out)
        if use_xm:
          x_t_out = torch.where(mask_me, x_m, x_t_out)
        
        return x_m, x_t_out

    def __init__(self,thresh_e=0.3, thresh_m=0.3, save_inter = False):
        self.thresh_e = thresh_e
        self.thresh_m = thresh_m
        self.save_inter = save_inter
        
    def set_map(self, ms, alpha, alpha_e, alpha_m,len):
        self.m = ms
        self.alpha = alpha
        self.alpha_e = alpha_e
        self.alpha_m = alpha_m
        alpha_me = alpha_e.to(torch.bool) & alpha_m.to(torch.bool)
        self.alpha_me = alpha_me.to(torch.float)
        self.len = len


class AttentionControl(abc.ABC):

    def step_callback(self, x_t):
        return x_t

    def between_steps(self):
        return

    @property
    def num_uncond_att_layers(self):
        return self.num_att_layers if LOW_RESOURCE else 0

    @abc.abstractmethod
    def forward(self, attn, is_cross: bool, place_in_unet: str):
        raise NotImplementedError

    def __call__(self, attn, is_cross: bool, place_in_unet: str):
        if self.cur_att_layer >= self.num_uncond_att_layers:
            if LOW_RESOURCE:
                attn = self.forward(attn, is_cross, place_in_unet)
            else:
                h = attn.shape[0]
                attn[h // 2:] = self.forward(attn[h // 2:], is_cross, place_in_unet)
        self.cur_att_layer += 1
        if self.cur_att_layer == self.num_att_layers // 2 + self.num_uncond_att_layers:
            self.cur_att_layer = 0
            self.cur_step += 1
            self.between_steps()
        return attn

    def reset(self):
        self.cur_step = 0
        self.cur_att_layer = 0

    def __init__(self):
        self.cur_step = 0
        self.num_att_layers = -1
        self.cur_att_layer = 0


class EmptyControl(AttentionControl):

    def forward(self, attn, is_cross: bool, place_in_unet: str):
        return attn

    def self_attn_forward(self, q, k, v, num_heads):
        # No UAC injection — return q/k/v unchanged so attention runs normally
        return q, k, v


class AttentionStore(AttentionControl):

    @staticmethod
    def get_empty_store():
        return {"down_cross": [], "mid_cross": [], "up_cross": [],
                "down_self": [],  "mid_self": [],  "up_self": []}

    def forward(self, attn, is_cross: bool, place_in_unet: str):
        key = f"{place_in_unet}_{'cross' if is_cross else 'self'}"
        if attn.shape[1] <= 33 ** 2:
            self.step_store[key].append(attn)
        return attn

    def between_steps(self):
        if len(self.attention_store) == 0:
            self.attention_store = self.step_store
        else:
            for key in self.attention_store:
                for i in range(len(self.attention_store[key])):
                    self.attention_store[key][i] += self.step_store[key][i]
        self.step_store = self.get_empty_store()

    def get_average_attention(self):
        average_attention = {key: [item / self.cur_step for item in self.attention_store[key]] for key in self.attention_store}
        return average_attention

    def get(self, key, default=None):
        """Get attention maps by key (e.g., 'down_cross', 'up_cross')."""
        return self.attention_store.get(key, default)

    def reset(self):
        super(AttentionStore, self).reset()
        self.step_store = self.get_empty_store()
        self.attention_store = {}

    def __init__(self):
        super(AttentionStore, self).__init__()
        self.step_store = self.get_empty_store()
        self.attention_store = {}


class AttentionControlEdit(AttentionStore, abc.ABC):

    def step_callback(self,i, t, x_s, x_t, x_m, alpha_prod):
        if (self.local_blend is not None) and (i>0):
            use_xm = (self.cur_step+self.start_steps+1 == self.num_steps)
            x_m, x_t = self.local_blend(i, x_s, x_t, x_m, self.attention_store, alpha_prod, use_xm=use_xm)
        return x_m, x_t

    def replace_self_attention(self, attn_base, att_replace):
        if att_replace.shape[2] <= 16 ** 2:
            return attn_base.unsqueeze(0).expand(att_replace.shape[0], *attn_base.shape)
        else:
            return att_replace

    @abc.abstractmethod
    def replace_cross_attention(self, attn_base, att_replace):
        raise NotImplementedError
    
    def attn_batch(self, q, k, v, sim, attn, is_cross, place_in_unet, num_heads, **kwargs):
        b = q.shape[0] // num_heads

        sim = torch.einsum("h i d, h j d -> h i j", q, k) * kwargs.get("scale")
        attn = sim.softmax(-1)
        out = torch.einsum("h i j, h j d -> h i d", attn, v)
        return out
    
    def self_attn_forward(self, q, k, v, num_heads):
        num_streams = q.shape[0] // num_heads
        past_replace = (self.self_replace_steps <= ((self.cur_step + self.start_steps + 1) * 1.0 / self.num_steps))

        if _backbone == "uvit" and num_streams == 3:
            # U-ViT non-CFG path: pipeline sends 3 streams [source, target, mutual].
            # Each stream is num_heads rows.
            q_s = q[:num_heads]           # source
            q_t = q[num_heads:num_heads*2]  # target
            q_m = q[num_heads*2:]           # mutual
            k_s = k[:num_heads]
            k_t = k[num_heads:num_heads*2]
            k_m = k[num_heads*2:]
            v_s = v[:num_heads]
            v_t = v[num_heads:num_heads*2]
            v_m = v[num_heads*2:]
            if past_replace:
                # Inject source structure into target and mutual streams
                k_t = k_s
                v_t = v_s
                k_m = k_s
                v_m = v_s
            return (torch.cat([q_s, q_t, q_m]),
                    torch.cat([k_s, k_t, k_m]),
                    torch.cat([v_s, v_t, v_m]))

        elif _backbone == "uvit" and num_streams == 6:
            # U-ViT CFG path: 6 streams [src_unc, tgt_unc, mut_unc, src_cond, tgt_cond, mut_cond].
            H = num_heads
            q_su, q_tu, q_mu = q[:H], q[H:H*2], q[H*2:H*3]
            q_sc, q_tc, q_mc = q[H*3:H*4], q[H*4:H*5], q[H*5:]
            k_su, k_tu, k_mu = k[:H], k[H:H*2], k[H*2:H*3]
            k_sc, k_tc, k_mc = k[H*3:H*4], k[H*4:H*5], k[H*5:]
            v_su, v_tu, v_mu = v[:H], v[H:H*2], v[H*2:H*3]
            v_sc, v_tc, v_mc = v[H*3:H*4], v[H*4:H*5], v[H*5:]
            if past_replace:
                k_tu, v_tu = k_su, v_su
                k_mu, v_mu = k_su, v_su
                k_tc, v_tc = k_sc, v_sc
                k_mc, v_mc = k_sc, v_sc
            return (torch.cat([q_su, q_tu, q_mu, q_sc, q_tc, q_mc]),
                    torch.cat([k_su, k_tu, k_mu, k_sc, k_tc, k_mc]),
                    torch.cat([v_su, v_tu, v_mu, v_sc, v_tc, v_mc]))

        elif num_streams == 3:
            # U-Net 3-stream path
            if past_replace:
                q=torch.cat([q[:num_heads*2],q[num_heads:num_heads*2]])
                k=torch.cat([k[:num_heads*2],k[:num_heads]])
                v=torch.cat([v[:num_heads*2],v[:num_heads]])
            else:
                q=torch.cat([q[:num_heads],q[:num_heads],q[:num_heads]])
                k=torch.cat([k[:num_heads],k[:num_heads],k[:num_heads]])
                v=torch.cat([v[:num_heads*2],v[:num_heads]])
            return q, k, v

        else:
            # U-Net standard CFG path: 4+ streams
            qu, qc = q.chunk(2)
            ku, kc = k.chunk(2)
            vu, vc = v.chunk(2)
            if past_replace:
                qu=torch.cat([qu[:num_heads*2],qu[num_heads:num_heads*2]])
                qc=torch.cat([qc[:num_heads*2],qc[num_heads:num_heads*2]])
                ku=torch.cat([ku[:num_heads*2],ku[:num_heads]])
                kc=torch.cat([kc[:num_heads*2],kc[:num_heads]])
                vu=torch.cat([vu[:num_heads*2],vu[:num_heads]])
                vc=torch.cat([vc[:num_heads*2],vc[:num_heads]])
            else:
                qu=torch.cat([qu[:num_heads],qu[:num_heads],qu[:num_heads]])
                qc=torch.cat([qc[:num_heads],qc[:num_heads],qc[:num_heads]])
                ku=torch.cat([ku[:num_heads],ku[:num_heads],ku[:num_heads]])
                kc=torch.cat([kc[:num_heads],kc[:num_heads],kc[:num_heads]])
                vu=torch.cat([vu[:num_heads*2],vu[:num_heads]])
                vc=torch.cat([vc[:num_heads*2],vc[:num_heads]])
            return torch.cat([qu, qc], dim=0), torch.cat([ku, kc], dim=0), torch.cat([vu, vc], dim=0)

    def forward(self, attn, is_cross: bool, place_in_unet: str):
        if is_cross:
            h = attn.shape[0] // self.batch_size
            attn = attn.reshape(self.batch_size, h, *attn.shape[1:])
            # Three streams: [source, target, mutual/masa] — same for UNet and UViT
            attn_base, attn_replace, attn_masa = attn[0], attn[1], attn[2]
            attn_replace_new = self.replace_cross_attention(attn_masa, attn_replace)
            attn_base_store = self.replace_cross_attention(attn_base, attn_replace)
            if (self.cross_replace_steps >= ((self.cur_step + self.start_steps + 1) * 1.0 / self.num_steps)):
                attn[1] = attn_replace_new
            attn_store = torch.cat([attn_base_store, attn_replace_new])
            attn = attn.reshape(self.batch_size * h, *attn.shape[2:])
            attn_store = attn_store.reshape(2 * h, *attn_store.shape[2:])
            super(AttentionControlEdit, self).forward(attn_store, is_cross, place_in_unet)
        return attn

    def __init__(self, prompts, num_steps: int,start_steps: int,
                 cross_replace_steps: Union[float, Tuple[float, float], Dict[str, Tuple[float, float]]],
                 self_replace_steps: Union[float, Tuple[float, float]],
                 local_blend: Optional[LocalBlend]):
        super(AttentionControlEdit, self).__init__()
        # Pipeline always sends 3 streams: source + target + mutual (masa)
        # This is true for both U-Net and U-ViT backends
        self.batch_size = len(prompts) + 1
        self.self_replace_steps = self_replace_steps
        self.cross_replace_steps = cross_replace_steps
        self.num_steps=num_steps
        self.start_steps=start_steps
        self.local_blend = local_blend


class AttentionReplace(AttentionControlEdit):

    def replace_cross_attention(self, attn_base, att_replace):
        return torch.einsum('hpw,bwn->bhpn', attn_base, self.mapper)

    def __init__(self, prompts, num_steps: int, cross_replace_steps: float, self_replace_steps: float,
                 local_blend: Optional[LocalBlend] = None):
        super(AttentionReplace, self).__init__(prompts, num_steps, cross_replace_steps, self_replace_steps, local_blend)
        self.mapper = seq_aligner.get_replacement_mapper(prompts, tokenizer).to(device).to(torch_dtype)


class AttentionRefine(AttentionControlEdit):

    def replace_cross_attention(self, attn_masa, att_replace):
        attn_masa_replace = attn_masa[:, :, self.mapper].squeeze()
        attn_replace = attn_masa_replace * self.alphas + \
                 att_replace * (1 - self.alphas)
        return attn_replace

    def __init__(self, prompts, prompt_specifiers, num_steps: int,start_steps: int, cross_replace_steps: float, self_replace_steps: float,
                 local_blend: Optional[LocalBlend] = None):
        super(AttentionRefine, self).__init__(prompts, num_steps,start_steps, cross_replace_steps, self_replace_steps, local_blend)
        self.mapper, alphas, ms, alpha_e, alpha_m = seq_aligner.get_refinement_mapper(prompts, prompt_specifiers, tokenizer, encoder, device)
        self.mapper, alphas, ms = self.mapper.to(device), alphas.to(device).to(torch_dtype), ms.to(device).to(torch_dtype)
        self.alphas = alphas.reshape(alphas.shape[0], 1, 1, alphas.shape[1])
        self.ms = ms.reshape(ms.shape[0], 1, 1, ms.shape[1])
        ms = ms.to(device)
        alpha_e = alpha_e.to(device)
        alpha_m = alpha_m.to(device)
        t_len = len(tokenizer(prompts[1])["input_ids"])
        self.local_blend.set_map(ms,alphas,alpha_e,alpha_m,t_len)


def get_equalizer(text: str, word_select: Union[int, Tuple[int, ...]], values: Union[List[float], Tuple[float, ...]]):
    if type(word_select) is int or type(word_select) is str:
        word_select = (word_select,)
    equalizer = torch.ones(len(values), 77)
    values = torch.tensor(values, dtype=torch_dtype)
    for word in word_select:
        inds = ptp_utils.get_word_inds(text, word, tokenizer)
        equalizer[:, inds] = values
    return equalizer

def _has_processors_on_unet(unet):
    # For UViT, use the dedicated has_attention_processors check which tests
    # for non-None processors rather than just attribute existence.
    if _backbone == "uvit":
        try:
            return has_attention_processors(unet)
        except Exception:
            return False
    # Original U-Net check
    for m in unet.modules():
        if hasattr(m, "processor") and getattr(m, "processor") is not None:
            return True
    return False


def inference(source_prompt, target_prompt, positive_prompt, negative_prompt, local, mutual, guidance_s, guidance_t, num_inference_steps=10,
              width=512, height=512, seed=0, img=None, strength=0.7,
               cross_replace_steps=0.8, self_replace_steps=0.4, eta=0.1, thresh_e=0.3, thresh_m=0.3, denoise=True):

    torch.manual_seed(seed)
    img = img.resize((width, height), Image.Resampling.LANCZOS)
    if denoise is False:
        strength = 1
    num_denoise_num = math.trunc(num_inference_steps*strength)
    num_start = num_inference_steps-num_denoise_num
    # create the CAC controller.
    local_blend = LocalBlend(thresh_e=thresh_e, thresh_m=thresh_m, save_inter=False)
    controller = AttentionRefine([source_prompt, target_prompt],[[local, mutual]],
                    num_inference_steps,
                    num_start,
                    cross_replace_steps=cross_replace_steps,
                    self_replace_steps=self_replace_steps,
                    local_blend=local_blend
                    )
    ptp_utils.register_attention_control(pipe, controller)
    if _backbone == "uvit":
        uvit_register(pipe.unet, controller)

    # sanity: ensure attention processors attached to the model (UViT or UNet)
    if not _has_processors_on_unet(pipe.unet):
        raise RuntimeError("No attention processors attached to the pipeline unet; UAC may not be active.")

    results = pipe(prompt=target_prompt,
                   source_prompt=source_prompt,
                   positive_prompt=positive_prompt,
                   negative_prompt=negative_prompt,
                   image=img,
                   num_inference_steps=num_inference_steps,
                   eta=eta,
                   strength=strength,
                   guidance_scale=guidance_t,
                   source_guidance_scale=guidance_s,
                   denoise_model=denoise,
                   callback = controller.step_callback
                   )

    return results.images[0]


def replace_nsfw_images(results):
    for i in range(len(results.images)):
        if results.nsfw_content_detected[i]:
            results.images[i] = Image.open("nsfw.png")
    return results.images[0]





def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source_path', type=str, required=True)
    parser.add_argument('--target_path', type=str, required=True)
    # inference parameter overrides (optional)
    parser.add_argument('--num_inference_steps', type=int, default=12)
    parser.add_argument('--strength', type=float, default=1.0)
    parser.add_argument('--cross_replace_steps', type=float, default=0.7)
    parser.add_argument('--self_replace_steps', type=float, default=0.7)
    parser.add_argument('--eta', type=float, default=1.0)
    parser.add_argument('--thresh_e', type=float, default=0.55)
    parser.add_argument('--thresh_m', type=float, default=0.6)
    parser.add_argument('--denoise', action='store_true', help='Run denoise mode (if not set uses denoise=False)')
    parser.add_argument('--seed', type=int, default=0, help='Random seed for inference')
    parser.add_argument('--guidance_t', type=float, default=2.3, help='Target guidance scale (higher = stronger text-driven editing)')
    parser.add_argument('--guidance_s', type=float, default=1.0, help='Source guidance scale')
    parser.add_argument('--backbone', type=str, default='unet', choices=['unet', 'uvit'])
    parser.add_argument('--uvit_size', type=str, default='mid', choices=['small', 'mid', 'large'])
    parser.add_argument('--uvit_checkpoint', type=str, default=None)
    parser.add_argument('--uvit_patch_size', type=int, default=2, help='Patch size used by the U-ViT checkpoint')
    parser.add_argument('--source_conditioned', action='store_true', help='Enable source-conditioned U-ViT; must match training/init')
    parser.add_argument('--patch_smooth_sigma', type=float, default=0.7,
                        help='Gaussian sigma (in latent pixels) applied to UViT predictions at inference to suppress '
                             '2-pixel patch-boundary artifacts. 0 disables. Default 0.7 attenuates 2px artifacts '
                             'to ~15%% while preserving 8px+ structure above 90%%.')

    args = parser.parse_args()

    # CLI is the source of truth for this run. Do not rely on BACKBONE / UVIT_* env vars.
    global _backbone, uvit_register, has_attention_processors
    _backbone = args.backbone

    if args.backbone == "uvit":
        from uvit_adapter import create_uvit_adapter, register_attention_control as uvit_register, has_attention_processors
        
        # If checkpoint provided, infer model size from it
        uvit_overrides = {
            'patch_size': args.uvit_patch_size,
            'source_conditioned': args.source_conditioned,
        }
        
        if args.uvit_checkpoint:
            raw = torch.load(args.uvit_checkpoint, map_location="cpu")
            state_dict = raw.get("model", raw.get("model_state_dict", raw))
            # torch.compile saves keys prefixed with "_orig_mod." — strip it
            if any(k.startswith("_orig_mod.") for k in state_dict):
                state_dict = {k.replace("_orig_mod.", "", 1): v for k, v in state_dict.items()}

            # Infer embed_dim from checkpoint
            if 'in_blocks.0.norm1.weight' in state_dict:
                embed_dim = state_dict['in_blocks.0.norm1.weight'].shape[0]
                print(f"Detected embed_dim={embed_dim} from checkpoint")
                uvit_overrides['embed_dim'] = embed_dim
            
            # Infer depth from checkpoint
            num_in_blocks = sum(1 for k in state_dict.keys() if k.startswith('in_blocks.') and '.norm1.weight' in k)
            num_out_blocks = sum(1 for k in state_dict.keys() if k.startswith('out_blocks.') and '.norm1.weight' in k)
            depth = num_in_blocks + 1 + num_out_blocks  # in + mid + out
            print(f"Detected depth={depth} from checkpoint")
            uvit_overrides['depth'] = depth
            
            # Infer img_size from pos_embed and patch_size
            if 'pos_embed' in state_dict and 'patch_embed.proj.weight' in state_dict:
                num_patches = state_dict['pos_embed'].shape[1] - 1  # subtract time token
                patch_size = state_dict['patch_embed.proj.weight'].shape[2]
                import math
                img_size = int(math.sqrt(num_patches) * patch_size)
                print(f"Detected img_size={img_size} from checkpoint")
                uvit_overrides['img_size'] = img_size
            
            # Infer in_chans from patch_embed
            if 'patch_embed.proj.weight' in state_dict:
                in_chans = state_dict['patch_embed.proj.weight'].shape[1]
                print(f"Detected in_chans={in_chans} from checkpoint")
                uvit_overrides['in_chans'] = in_chans
        
        _adapter = create_uvit_adapter(
            preset=args.uvit_size,
            patch_smooth_sigma=args.patch_smooth_sigma,
            **uvit_overrides
        )
        
        if args.uvit_checkpoint:
            missing, unexpected = _adapter.backbone.load_state_dict(state_dict, strict=False)
            print(f"Loaded UViT checkpoint from {args.uvit_checkpoint}")

            # Fail LOUD on any mismatch. Silent partial loads were causing inference
            # to run with random init in the unmatched layers, producing clean
            # reconstructions but no editing. If you see this error, something about
            # the checkpoint structure does not match the constructed UViT model.
            if missing or unexpected:
                print("=" * 70)
                print("CHECKPOINT LOAD MISMATCH — REFUSING TO RUN INFERENCE")
                print("=" * 70)
                if missing:
                    print(f"\n  Missing keys ({len(missing)}) — these MODEL params got random init:")
                    for k in missing[:20]:
                        print(f"    - {k}")
                    if len(missing) > 20:
                        print(f"    ... and {len(missing) - 20} more")
                if unexpected:
                    print(f"\n  Unexpected keys ({len(unexpected)}) — these CHECKPOINT params were dropped:")
                    for k in unexpected[:20]:
                        print(f"    - {k}")
                    if len(unexpected) > 20:
                        print(f"    ... and {len(unexpected) - 20} more")
                print("\n  Detected from checkpoint:")
                for k, v in uvit_overrides.items():
                    print(f"    {k} = {v}")
                print("\n  If this looks like a depth/embed_dim mismatch, the auto-detection")
                print("  in run_pie_bench.py needs to be extended. If keys are unexpectedly")
                print("  named (e.g. _orig_mod prefix), strip them before load_state_dict.")
                print("=" * 70)
                raise RuntimeError(
                    f"Refusing to run inference with partial weight load: "
                    f"{len(missing)} missing, {len(unexpected)} unexpected keys"
                )

            print(f"  -> Clean load: 0 missing, 0 unexpected keys.")
        _adapter = _adapter.to(dtype=torch_dtype)
        _adapter.eval()
        if torch.cuda.is_available():
            _adapter = _adapter.to("cuda")
        pipe.unet = _adapter

    root = args.source_path
    target = args.target_path
    

    annotation_file_name = os.path.join(root,"mapping_file.json")
    with open (annotation_file_name) as f:
        annotation_file = json.load(f)
    for annotation_idx , annotation  in annotation_file.items():
        print(annotation_idx)
        img_path =os.path.join(root, "annotation_images",annotation["image_path"] )
        # if os.path.exists( os.path.join(target, "annotation_images", annotation["image_path"])):
        #     continue
        imagein = Image.open(img_path)
        imagein = imagein.convert("RGB")
        source_prompt =  annotation["original_prompt"]
        target_prompt =  annotation["editing_prompt"]
        if annotation["blended_word"]!="":
            local = annotation["blended_word"].split(" ")[-1]
        else:
            local = ""
        # Source stream uses null text at inference to match training (source stream
        # was trained with empty-string conditioning, not the image description).
        image_out = inference(
            "", target_prompt, "", "", local, "", args.guidance_s, args.guidance_t,
            num_inference_steps=args.num_inference_steps,
            width=512, height=512, seed=args.seed, img=imagein,
            strength=args.strength,
            cross_replace_steps=args.cross_replace_steps,
            self_replace_steps=args.self_replace_steps,
            eta=args.eta,
            thresh_e=args.thresh_e,
            thresh_m=args.thresh_m,
            denoise=args.denoise)
        annotation_dir = os.path.dirname(annotation["image_path"])

        # Create the full directory path
        full_dir_path = os.path.join(target, "annotation_images", annotation_dir)
        os.makedirs(full_dir_path, exist_ok=True)

        # Mild Gaussian blur to reduce UViT patch-boundary artifacts (16px grid).
        # Sigma=0.8 softens seams without significantly blurring edit content.
        image_out = image_out.filter(ImageFilter.GaussianBlur(radius=0.8))

        out_path = os.path.join(full_dir_path, os.path.basename(annotation["image_path"]))
        image_out.save(out_path)    
    # Now you can use args.cross_replace_steps, args.guidance, and args.strength in your script

if __name__ == "__main__":
    main()