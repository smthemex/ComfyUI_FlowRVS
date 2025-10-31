# !/usr/bin/env python
# -*- coding: UTF-8 -*-
import os
import torch
import gc
from PIL import Image
import numpy as np
import cv2
from comfy.utils import common_upscale

cur_path = os.path.dirname(os.path.abspath(__file__))

FlowRVS_SM_origin_dict={
            "input_path":"",
            "text_prompts":"",
            "num_steps":4,
            "fps":24,
            "cfg":True,
            "save_fig":False,
            "seed":42,
            "device":"cuda",
            "model_id":"Wan2.1-T2V-1.3B-Diffusers",#wan2.1/Wan2.1-T2V-14B-Diffusers
            "local_dir":cur_path,
            "lr":7e-5,
            "lr_drop":30,
            "batch_size":1,
            "weight_decay":5e-4,
            "epochs":200,
            "clip_max_norm":0.1,
            "amp":False,
            "resume":"", #PATH
            "output_dir":"",
            "sets":"val",
            "task":"unsupervised",#'semi-supervised'
            "num_workers":4,
            "cache_mode":False,
            "world_size":1,
            "dist_url":'env://',
            "pretrained_weights":"",
            "freeze_text_encoder":True,
            "freeze_transformer_core":True,
            "use_lora":False,
            "lora_rank":4,
            "enable_gradient_checkpointing":True,
            "mask_latent_channels":4,
            "lr_lora":1e-4,
            "lr_mask_encoder":None,
            "lr_mask_decoder":None,
            "video_latent_loss_weight":1.0,
            "mask_latent_flow_loss_weight":1.0,
            "mask_pixel_focal_loss_weight":8.0,
            "mask_pixel_dice_loss_weight":4.0,
            "focal_alpha":0.25,
            "num_frames":17,
            "image_size":512,
            "latent_resolution_scale":8,
            "dataset_files":"ytvos",
            "ytvos_path":"datasets/refer_youtube_vos",
            "davis_path":"datasets/refer_davis",
            "a2d_path":"datasets/a2d_sentences",
            "jhmdb_path":"datasets/jhmdb_sentences",
            "mevis_path":"datasets/MeViS",
            "coco_path":"datasets/coco",
            "augm_resize":True,
            "max_skip":3,
            "max_size":832,
            "binary":True,
            "remove_difficult":True,
            "eval":True,
            "threshold":0.5,
            "ngpu":1,
            "split":"valid",  #'test' 'valid_u', 'train'
            "visualize":True,
            "tag":"debug",
            "exp_name":"main",
            "current_epoch":0,
            "high_reso":False, #5B
            "big":False, #5B
        }
def gc_cleanup():
    gc.collect()
    torch.cuda.empty_cache()

def tensor2cv(tensor_image):
    if len(tensor_image.shape)==4:# b hwc to hwc
        tensor_image=tensor_image.squeeze(0)
    if tensor_image.is_cuda:
        tensor_image = tensor_image.cpu()
    tensor_image=tensor_image.numpy()
    #反归一化
    maxValue=tensor_image.max()
    tensor_image=tensor_image*255/maxValue
    img_cv2=np.uint8(tensor_image)#32 to uint8
    img_cv2=cv2.cvtColor(img_cv2,cv2.COLOR_RGB2BGR)
    return img_cv2

def phi2narry(img):
    img = torch.from_numpy(np.array(img).astype(np.float32) / 255.0).unsqueeze(0)
    return img

def tensor2image(tensor):
    image_np = tensor.squeeze().mul(255).clamp(0, 255).byte().numpy()
    image = Image.fromarray(image_np, mode='RGB')
    return image

def tensor2pillist(tensor_in):
    d1, _, _, _ = tensor_in.size()
    if d1 == 1:
        img_list = [tensor2image(tensor_in)]
    else:
        tensor_list = torch.chunk(tensor_in, chunks=d1)
        img_list=[tensor2image(i) for i in tensor_list]
    return img_list

def tensor2pillist_upscale(tensor_in,width,height):
    d1, _, _, _ = tensor_in.size()
    if d1 == 1:
        img_list = [nomarl_upscale(tensor_in,width,height)]
    else:
        tensor_list = torch.chunk(tensor_in, chunks=d1)
        img_list=[nomarl_upscale(i,width,height) for i in tensor_list]
    return img_list

def tensor2list(tensor_in,width,height):
    if tensor_in is None:
        return None
    d1, _, _, _ = tensor_in.size()
    if d1 == 1:
        tensor_list = [tensor_upscale(tensor_in,width,height)]
    else:
        tensor_list_ = torch.chunk(tensor_in, chunks=d1)
        tensor_list=[tensor_upscale(i,width,height) for i in tensor_list_]
    return tensor_list


def tensor_upscale(tensor, width, height):
    samples = tensor.movedim(-1, 1)
    samples = common_upscale(samples, width, height, "nearest-exact", "center")
    samples = samples.movedim(1, -1)
    return samples

def nomarl_upscale(img, width, height):
    samples = img.movedim(-1, 1)
    img = common_upscale(samples, width, height, "nearest-exact", "center")
    samples = img.movedim(1, -1)
    img = tensor2image(samples)
    return img



def cv2tensor(img,bgr2rgb=True):
    assert type(img) == np.ndarray, 'the img type is {}, but ndarry expected'.format(type(img))
    if bgr2rgb:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = torch.from_numpy(img.transpose((2, 0, 1)))
    return img.float().div(255).permute(1, 2, 0).unsqueeze(0)  




