import os
import sys
import argparse
import random
import time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw
from tqdm import tqdm

#import opts
from .util import misc  as utils_
from .models.wan_rvos import build_dit
from .models.text import TextProcessor
from diffusers.models import AutoencoderKLWan
from diffusers.schedulers import FlowMatchEulerDiscreteScheduler
from transformers import AutoTokenizer, UMT5EncoderModel
from .models.mask_vae_finetuner import MaskVAEFinetuner
from .datasets.transform_utils import VideoEvalDataset, vis_add_mask_new, check_shape
from .utils_inf import colormap
from torch.utils.data import DataLoader
from moviepy import ImageSequenceClip
from comfy.utils import common_upscale
import comfy.model_management as mm
import cv2
from scipy import ndimage



opts={}
color_list = colormap().astype('uint8').tolist()


def tensor_upscale(tensor, width, height):
    samples = tensor.movedim(-1, 1)
    samples = common_upscale(samples, width, height, "nearest-exact", "center")
    samples = samples.movedim(1, -1)
    return samples

def extract_frames_from_mp4(video_path, output_folder):

    needs_extraction = True
    if os.path.isdir(output_folder) and len(os.listdir(output_folder)) > 0:
        needs_extraction = False
        print(f'{output_folder} exists')

    if needs_extraction:
        os.makedirs(output_folder, exist_ok=True) 

        extract_cmd = f"ffmpeg -i \"{video_path}\" -loglevel error -vf fps={args.fps} \"{output_folder}/frame_%05d.png\""
        ret = os.system(extract_cmd)
        if ret != 0:
            if len(os.listdir(output_folder)) == 0:
                os.rmdir(output_folder)
            sys.exit(ret)
            
    frames_list = sorted([os.path.splitext(f)[0] for f in os.listdir(output_folder) if f.endswith('.png')])
    return output_folder, frames_list, '.png'


def prepare_models(args):
    device = torch.device(args.device)

    model = build_dit(args)
    model_id= args.model_id
    #local_repo= os.path.join(args.local_dir, model_id)
    #model_id = "Wan2.1-T2V-1.3B-Diffusers" #
   

    #model_id = "Wan2.1-T2V-1.3B-Diffusers"   
    mask_vae = MaskVAEFinetuner(vae_model_id=model_id,vae_path=args.vae_path, target_dtype=torch.bfloat16)
    
    #print(f"[Loading checkpoint from {args.vae_ckpt}]")
    vae_checkpoint = torch.load(args.vae_ckpt, map_location='cpu', weights_only=False)
    vae_state_dict = vae_checkpoint.get('model', vae_checkpoint)
    #torch.save(vae_state_dict, 'decoder.pth')
    missing_keys, unexpected_keys = mask_vae.load_state_dict(vae_state_dict, strict=True)
    unexpected_keys = [k for k in unexpected_keys if not (k.endswith('total_params') or k.endswith('total_ops'))]
    if len(missing_keys) > 0:
        print(f'Missing Keys: {missing_keys}')
    if len(unexpected_keys) > 0:
        print(f'Unexpected Keys: {unexpected_keys}')
    del vae_checkpoint

    vae = mask_vae.vae.to(device).eval() 

    if args.resume:
        print(f"[Loading checkpoint from {args.resume}")
        checkpoint = torch.load(args.resume, map_location='cpu', weights_only=False)
        model_state_dict = checkpoint.get('model', checkpoint) 

        missing_keys, unexpected_keys = model.load_state_dict(model_state_dict, strict=False)

        unexpected_keys = [k for k in unexpected_keys if not (k.endswith('total_params') or k.endswith('total_ops'))]
        if len(missing_keys) > 0:
            print(f'Missing Keys: {missing_keys}')
        if len(unexpected_keys) > 0:
            print(f'Unexpected Keys: {unexpected_keys}')
        del checkpoint
    else:
        raise ValueError('Please specify the checkpoint for inference using --resume.')
    
    model.to(torch.bfloat16).eval() 

    return model, vae


def load_text_processor(model_id,text_encoder,device):
    tokenizer = AutoTokenizer.from_pretrained(model_id, subfolder="tokenizer")
    #text_encoder = UMT5EncoderModel.from_pretrained(model_id, subfolder="text_encoder")
    #text_processor = TextProcessor(tokenizer, text_encoder.to(device).eval())
    text_processor = TextProcessor(tokenizer, text_encoder)
    # text_processor.text_encoder.to(torch.bfloat16)
    return text_processor

def data_processor(prompt_embeds,vae,video_t,device,dtype):
   
    device, dtype = vae.device, vae.dtype
    mean_tensor = torch.tensor(vae.config.latents_mean, device=device, dtype=dtype).view(1, -1, 1, 1, 1)
    std_tensor = torch.tensor(vae.config.latents_std, device=device, dtype=dtype).view(1, -1, 1, 1, 1)


    target_h, target_w = (480, 832)
    t,origin_h,origin_w,_=video_t.shape
    imgs = tensor_upscale(video_t, target_w, target_h)
    imgs = imgs.to(device)

    if (t - 1) % 4 != 0:
        num_padding_frames = (4 - (t - 1) % 4) % 4
        padding_frames = imgs[-1:].repeat(num_padding_frames, 1, 1, 1)
        imgs = torch.cat([imgs, padding_frames], dim=0)
    
    imgs = imgs.unsqueeze(0).permute(0,4,1,2,3) # 1 T H W C  --> [B, C_mask, F, H_video, W_video]
    # print(imgs.shape) #torch.Size([1, 3, 109, 480, 832])
    with torch.no_grad():
        #prompt_embeds, _ = text_processor.encode_prompt_and_cfg(prompt=[prompt], device=device, dtype=dtype, do_classifier_free_guidance=cfg)
        #del text_processor
        
        imgs = check_shape(imgs)
        x0_video_latent = vae.encode(imgs.to(vae.dtype)).latent_dist.mean
        #x0_video_latent = vae.encode(imgs.transpose(1, 2).to(vae.dtype)).latent_dist.mean
        x0_video_latent = (x0_video_latent - mean_tensor) / std_tensor
        x0_video_latent=x0_video_latent.to(device)
        prompt_embeds=prompt_embeds.to(device,dtype=dtype)
    vae.to("cpu")
    torch.cuda.empty_cache()
    cond={"prompt_embeds": prompt_embeds,"x0_video_latent": x0_video_latent,}
    param={"original_len": t, "origin_h": origin_h, "origin_w": origin_w}
    return cond,param

def inference_single_video(model,steps, model_id, x0_video_latent, prompt_embeds,device):
    model=model.to(device)
    scheduler = FlowMatchEulerDiscreteScheduler.from_config(model_id, subfolder="scheduler")

    # fname, ext = os.path.splitext(os.path.basename(video_path))
    # if ext.lower() == '.mp4':
    #     temp_frames_folder = os.path.join(args.output_dir, f"frames_{fname}")
    #     frames_folder, frames_list, frame_ext = extract_frames_from_mp4(video_path, temp_frames_folder)
    # elif os.path.isdir(video_path):
    #     frames_folder = video_path
    #     all_files = os.listdir(frames_folder)
    #     image_files = [f for f in all_files if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    #     if not image_files:
    #         frames_list = []
    #         frame_ext = None
    #     else:
    #         frames_list = sorted([os.path.splitext(f)[0] for f in image_files])
    #         frame_ext = os.path.splitext(image_files[0])[1]
    # else:
    #     raise ValueError(f"Path is not rigjt: {video_path}")
 
    # device, dtype = vae.device, vae.dtype
    # mean_tensor = torch.tensor(vae.config.latents_mean, device=device, dtype=dtype).view(1, -1, 1, 1, 1)
    # std_tensor = torch.tensor(vae.config.latents_std, device=device, dtype=dtype).view(1, -1, 1, 1, 1)

    # target_h, target_w = (480, 832)
    # total,origin_h,origin_w,_=video_t.shape
    # imgs = tensor_upscale(video_t, target_w, target_h)
    #vd = VideoEvalDataset(frames_folder, frames_list, frame_ext, target_h=target_h, target_w=target_w)
    #dl = DataLoader(vd, batch_size=len(frames_list), num_workers=args.num_workers, shuffle=False)
    #origin_w, origin_h = vd.origin_w, vd.origin_h

    #(imgs, _) = next(iter(dl))
    # imgs = imgs.to(device)

    # t = imgs.shape[0]

    # original_len = t
    # if (t - 1) % 4 != 0:
    #     num_padding_frames = (4 - (t - 1) % 4) % 4
    #     padding_frames = imgs[-1:].repeat(num_padding_frames, 1, 1, 1)
    #     imgs = torch.cat([imgs, padding_frames], dim=0)
    latents_mean= [
        -0.7571,
        -0.7089,
        -0.9113,
        0.1075,
        -0.1745,
        0.9653,
        -0.1517,
        1.5508,
        0.4134,
        -0.0715,
        0.5517,
        -0.3632,
        -0.1922,
        -0.9497,
        0.2503,
        -0.2921
        ]
    latents_std= [
        2.8184,
        1.4541,
        2.3275,
        2.6558,
        1.2196,
        1.7708,
        2.6052,
        2.0743,
        3.2687,
        2.1526,
        2.8652,
        1.5579,
        1.6382,
        1.1253,
        2.8251,
        1.916
        ]
    # imgs = imgs.unsqueeze(0).permute(0,4,1,2,3) # 1 T H W C  --> [B, C_mask, F, H_video, W_video]
    latents_mean = torch.tensor(latents_mean).view(1, 16, 1, 1, 1)
    latents_std = torch.tensor(latents_std).view(1, 16, 1, 1, 1)
    mean_tensor = latents_mean.to(device=device, dtype=model.dtype)
    std_tensor = latents_std.to(device=device, dtype=model.dtype)
    with torch.no_grad():
        # imgs = check_shape(imgs)
        # x0_video_latent = vae.encode(imgs.transpose(1, 2).to(vae.dtype)).latent_dist.mean
        # x0_video_latent = (x0_video_latent - mean_tensor) / std_tensor

        # for i, prompt in enumerate(tqdm(text_prompts, desc=f"Processing prompts for {fname}")):
            # prompt_embeds, _ = text_processor.encode_prompt_and_cfg(
            #     prompt=[prompt], device=device, dtype=dtype, do_classifier_free_guidance=args.cfg
            # )

        shift = 3
        t_steps = torch.linspace(1.0, 0.001, steps + 1, device=device)
        timesteps = shift * t_steps / (1 + (shift - 1) * t_steps) * 1000
        scheduler.set_timesteps(num_inference_steps=steps, device=device)
        timesteps = scheduler.timesteps

        latents = x0_video_latent.clone()
        for t in tqdm(timesteps, leave=False, desc="Diffusion steps"):
            timestep = t.expand(latents.shape[0])
            noise_pred = model(
                hidden_states=latents.to(model.dtype),
                video_condition=x0_video_latent.to(model.dtype),
                timestep=timestep,
                encoder_hidden_states=prompt_embeds,
            )[0]
            latents = scheduler.step(noise_pred, t, latents)[0]
        latents = latents * std_tensor + mean_tensor
    model.to("cpu")
    torch.cuda.empty_cache()
    #print('latents', latents.shape) #latents torch.Size([1, 16, 28, 60, 104])
    return latents

def decode_latents(vae,latents,origin_h, origin_w,original_len,threshold,device,morphological,connected_components,gaussian,shrink_pixels=0,kernel_size=3,min_area_ratio=0.01,sigma=1.0,shrink_method="uniform"):
    #latents = latents * std_tensor_mask + mean_tensor_mask
    vae=vae.to(device)
    decoded_pixel_output = vae.decode(latents.detach())[0]
    decoded_pixel_output = F.interpolate(decoded_pixel_output.view(-1, 1, decoded_pixel_output.shape[-2], decoded_pixel_output.shape[-1]),
                        size=(origin_h, origin_w), mode='bilinear', align_corners=False) 
    reconstructed_mask_probs = torch.sigmoid(decoded_pixel_output)
    all_pred_masks = (reconstructed_mask_probs > threshold).float().cpu().squeeze(0).squeeze(1)
    #all_pred_masks = all_pred_masks[:original_len] 
    all_pred_masks = all_pred_masks[:original_len].numpy() 
    #print('all_pred_masks', all_pred_masks.shape) #all_pred_masks (106, 320, 512) 
    if morphological:
        all_pred_masks = morphological_postprocessing(all_pred_masks, kernel_size=kernel_size)
    if connected_components:
        all_pred_masks = connected_components_filter(all_pred_masks, min_area_ratio=min_area_ratio)
    if gaussian:
        all_pred_masks = gaussian_smoothing(all_pred_masks, sigma=sigma)
    if shrink_pixels > 0:
        all_pred_masks = shrink_mask_center_advanced(all_pred_masks, shrink_pixels, shrink_method)
    if isinstance(all_pred_masks, np.ndarray):
        all_pred_masks = torch.from_numpy(all_pred_masks)
    return all_pred_masks


def morphological_postprocessing(mask, kernel_size=3):
    if isinstance(mask, torch.Tensor):
        mask_np = mask.cpu().numpy()
    else:
        mask_np = mask
    original_shape = mask_np.shape
    if len(original_shape) == 3:
        processed_masks = []
        for i in range(original_shape[0]):
            single_mask = mask_np[i]
            processed_mask = morphological_single_mask(single_mask, kernel_size)
            processed_masks.append(processed_mask)
        return np.stack(processed_masks, axis=0)
    else:
        return morphological_single_mask(mask_np, kernel_size)

def morphological_single_mask(mask, kernel_size=3):
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    binary_mask = (mask > 0.5).astype(np.uint8)
    opened = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel)
    closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel)
    dilated = cv2.dilate(closed, kernel, iterations=1)
    return dilated.astype(np.float32)

def connected_components_filter(mask, min_area_ratio=0.01):
    if isinstance(mask, torch.Tensor):
        mask_np = mask.cpu().numpy()
    else:
        mask_np = mask
    
    original_shape = mask_np.shape
    if len(original_shape) == 3:
        processed_masks = []
        for i in range(original_shape[0]):
            single_mask = mask_np[i]
            processed_mask = connected_components_single(single_mask, min_area_ratio)
            processed_masks.append(processed_mask)
        result = np.stack(processed_masks, axis=0)
    else:
        result = connected_components_single(mask_np, min_area_ratio)
    
    if isinstance(mask, torch.Tensor):
        return torch.from_numpy(result).to(mask.device)
    return result

def connected_components_single(mask, min_area_ratio=0.01):

    binary_mask = (mask > 0.5).astype(np.uint8)

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary_mask, connectivity=8)
    
    if num_labels <= 1:
        return binary_mask.astype(np.float32)

    total_area = np.sum(binary_mask)
    min_area = total_area * min_area_ratio

    filtered_mask = np.zeros_like(binary_mask)
    for i in range(1, num_labels):  # 0是背景
        area = stats[i, cv2.CC_STAT_AREA]
        if area >= min_area:
            filtered_mask[labels == i] = 1
    
    return filtered_mask.astype(np.float32)

def gaussian_smoothing(mask, sigma=1.0):
    if isinstance(mask, torch.Tensor):
        mask_np = mask.cpu().numpy()
    else:
        mask_np = mask
    
    original_shape = mask_np.shape
    if len(original_shape) == 3:
        processed_masks = []
        for i in range(original_shape[0]):
            single_mask = mask_np[i]
            smoothed = ndimage.gaussian_filter(single_mask, sigma=sigma)
            processed_masks.append(smoothed)
        result = np.stack(processed_masks, axis=0)
    else:
        result = ndimage.gaussian_filter(mask_np, sigma=sigma)
    
    if isinstance(mask, torch.Tensor):
        return torch.from_numpy(result).to(mask.device)
    return result

def shrink_mask_center_advanced(mask, pixels, method="uniform"):

    if isinstance(mask, torch.Tensor):
        mask_np = mask.cpu().numpy()
    else:
        mask_np = mask
    
    original_shape = mask_np.shape
    if len(original_shape) == 3:
        processed_masks = []
        for i in range(original_shape[0]):
            single_mask = mask_np[i]
            processed_mask = shrink_single_mask_advanced(single_mask, pixels, method)
            processed_masks.append(processed_mask)
        return np.stack(processed_masks, axis=0)
    else:
        return shrink_single_mask_advanced(mask_np, pixels, method)

def shrink_single_mask_advanced(mask, pixels, method="uniform"):

    binary_mask = (mask > 0.5).astype(np.uint8)
    
    if pixels <= 0 or np.sum(binary_mask) == 0:
        return mask.astype(np.float32)
    
    if method == "uniform":
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (pixels*2+1, pixels*2+1))
        shrunk_mask = cv2.erode(binary_mask, kernel, iterations=1)
        
    elif method == "distance":

        dist_transform = cv2.distanceTransform(binary_mask, cv2.DIST_L2, 5)
        shrunk_mask = (dist_transform > pixels).astype(np.uint8)
    
    return shrunk_mask.astype(np.float32)


def process_video_tensor(video_tensor: torch.Tensor, value) -> tuple:
    """
    处理视频张量尺寸，使其符合 832/480 比例且能被 16 整除。

    参数:
        video_tensor (torch.Tensor): 输入视频张量，形状为 [1, H, W, C]
        value: 填充值（如0.0表示黑底，1.0表示白底）

    返回:
        tuple: (processed_tensor, original_info)
            - processed_tensor (torch.Tensor): 处理后的视频张量
            - original_info (dict): 包含原始尺寸和处理信息的字典
    """
    TARGET_RATIO = 832 / 480
    DIVISOR = 16
    _, orig_H, orig_W, C = video_tensor.shape
    
    # 保存原始信息
    original_info = {
        'original_height': orig_H,
        'original_width': orig_W,
        'padding_info': None,
        'crop_info': None
    }
    
    if abs(orig_W / orig_H - TARGET_RATIO) < 1e-6 and orig_W % DIVISOR == 0 and orig_H % DIVISOR == 0:
        return video_tensor, original_info

    if abs(orig_W / orig_H - TARGET_RATIO) < 1e-6:
        new_W = (orig_W // DIVISOR) * DIVISOR
        new_H = (orig_H // DIVISOR) * DIVISOR
        
        start_h = (orig_H - new_H) // 2
        start_w = (orig_W - new_W) // 2
        
        # 保存裁剪信息
        original_info['crop_info'] = {
            'start_h': start_h,
            'start_w': start_w,
            'crop_height': new_H,
            'crop_width': new_W
        }
        
        return video_tensor[:, start_h:start_h+new_H, start_w:start_w+new_W, :], original_info
    
    current_ratio = orig_W / orig_H
    
    if current_ratio > TARGET_RATIO:  # 宽度过大，需要增加高度
        target_H = int(orig_W / TARGET_RATIO)
        pad_top = (target_H - orig_H) // 2
        pad_bottom = target_H - orig_H - pad_top
        
        # 保存填充信息
        original_info['padding_info'] = {
            'pad_top': pad_top,
            'pad_bottom': pad_bottom,
            'pad_left': 0,
            'pad_right': 0,
            'pad_value': value
        }
        
        video_tensor_bchw = video_tensor.permute(0, 3, 1, 2)  # 1HWC -> 1CHW
        padded = F.pad(video_tensor_bchw, (0, 0, pad_top, pad_bottom), mode='constant', value=value)
        video_tensor = padded.permute(0, 2, 3, 1)  # 1CHW -> 1HWC
        
    elif current_ratio < TARGET_RATIO:  # 高度过大，需要增加宽度
        target_W = int(orig_H * TARGET_RATIO)
        pad_left = (target_W - orig_W) // 2
        pad_right = target_W - orig_W - pad_left
        
        # 保存填充信息
        original_info['padding_info'] = {
            'pad_top': 0,
            'pad_bottom': 0,
            'pad_left': pad_left,
            'pad_right': pad_right,
            'pad_value': value
        }
        
        video_tensor_bchw = video_tensor.permute(0, 3, 1, 2)  # 1HWC -> 1CHW
        padded = F.pad(video_tensor_bchw, (pad_left, pad_right, 0, 0), mode='constant', value=value)
        video_tensor = padded.permute(0, 2, 3, 1)  # 1CHW -> 1HWC

    _, H_new, W_new, _ = video_tensor.shape

    target_W_final = min(W_new, (W_new // DIVISOR) * DIVISOR)
    target_H_final = min(H_new, int(target_W_final / TARGET_RATIO))
    target_H_final = (target_H_final // DIVISOR) * DIVISOR

    if abs(target_W_final / target_H_final - TARGET_RATIO) > 1e-6:
        target_H_final = min(H_new, (H_new // DIVISOR) * DIVISOR)
        target_W_final = min(W_new, int(target_H_final * TARGET_RATIO))
        target_W_final = (target_W_final // DIVISOR) * DIVISOR

    start_h = (H_new - target_H_final) // 2
    start_w = (W_new - target_W_final) // 2
    
    # 保存裁剪信息
    original_info['crop_info'] = {
        'start_h': start_h,
        'start_w': start_w,
        'crop_height': target_H_final,
        'crop_width': target_W_final
    }
    
    return video_tensor[:, start_h:start_h+target_H_final, start_w:start_w+target_W_final, :], original_info

def restore_original_size(processed_tensor: torch.Tensor, original_info: dict, is_mask=False) -> torch.Tensor:
    """
    将处理后的张量恢复到原始尺寸
    
    参数:
        processed_tensor (torch.Tensor): 处理后的张量
        original_info (dict): process_video_tensor 返回的原始信息
        is_mask (bool): 是否为mask张量（BHW格式）
        
    返回:
        torch.Tensor: 恢复到原始尺寸的张量
    """
    orig_H = original_info['original_height']
    orig_W = original_info['original_width']
    padding_info = original_info['padding_info']
    crop_info = original_info['crop_info']
    
    # 如果没有进行任何处理，直接返回
    if padding_info is None and crop_info is None:
        return processed_tensor
    
    # 处理mask格式 (BHW)
    if is_mask:
        # 添加通道维度使其变为 BHWC
        if len(processed_tensor.shape) == 3:
            processed_tensor = processed_tensor.unsqueeze(-1)  # BHW -> BHWC
    
    # 逆向裁剪操作（如果进行了裁剪）
    if crop_info is not None:
        # 这里需要根据具体需求进行处理，因为裁剪是不可逆的
        # 可以通过插值将处理后的张量调整回裁剪前的尺寸
        _, proc_H, proc_W, _ = processed_tensor.shape
        if proc_H != crop_info['crop_height'] or proc_W != crop_info['crop_width']:
            # 如果尺寸不匹配，需要进行插值
            processed_tensor_bchw = processed_tensor.permute(0, 3, 1, 2)
            processed_tensor_bchw = F.interpolate(
                processed_tensor_bchw, 
                size=(crop_info['crop_height'], crop_info['crop_width']), 
                mode='bilinear', 
                align_corners=False
            )
            processed_tensor = processed_tensor_bchw.permute(0, 2, 3, 1)
    
    # 逆向填充操作（如果进行了填充）
    if padding_info is not None:
        pad_top = padding_info['pad_top']
        pad_bottom = padding_info['pad_bottom']
        pad_left = padding_info['pad_left']
        pad_right = padding_info['pad_right']
        
        # 移除填充部分
        if pad_top > 0 or pad_bottom > 0 or pad_left > 0 or pad_right > 0:
            _, H, W, _ = processed_tensor.shape
            processed_tensor = processed_tensor[:, pad_top:H-pad_bottom if pad_bottom > 0 else H, 
                                              pad_left:W-pad_right if pad_right > 0 else W, :]
    
    # 最后通过插值调整到精确的原始尺寸
    _, current_H, current_W, _ = processed_tensor.shape
    if current_H != orig_H or current_W != orig_W:
        processed_tensor_bchw = processed_tensor.permute(0, 3, 1, 2)
        processed_tensor_bchw = F.interpolate(
            processed_tensor_bchw, 
            size=(orig_H, orig_W), 
            mode='bilinear', 
            align_corners=False
        )
        processed_tensor = processed_tensor_bchw.permute(0, 2, 3, 1)
    
    # 如果是mask，恢复为BHW格式
    if is_mask and len(processed_tensor.shape) == 4 and processed_tensor.shape[-1] == 1:
        processed_tensor = processed_tensor.squeeze(-1)  # BHWC -> BHW
    
    return processed_tensor