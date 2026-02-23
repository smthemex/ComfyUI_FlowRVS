 # !/usr/bin/env python
# -*- coding: UTF-8 -*-

import numpy as np
import torch
import os
from omegaconf import OmegaConf
from .model_loader_utils import  FlowRVS_SM_origin_dict,gc_cleanup
from .FlowRVS.inference_demo import inference_single_video,data_processor,process_video_tensor,decode_latents,restore_original_size,clear_comfyui_cache,load_dit,load_vae_model
import folder_paths
from typing_extensions import override
from comfy_api.latest import ComfyExtension, io

MAX_SEED = np.iinfo(np.int32).max

device = torch.device(
    "cuda:0") if torch.cuda.is_available() else torch.device(
    "mps") if torch.backends.mps.is_available() else torch.device("cpu")

node_cr_path = os.path.dirname(os.path.abspath(__file__))

weigths_FlowRVS_current_path = os.path.join(folder_paths.models_dir, "FlowRVS")
if not os.path.exists(weigths_FlowRVS_current_path):
    os.makedirs(weigths_FlowRVS_current_path)
folder_paths.add_model_folder_path("FlowRVS", weigths_FlowRVS_current_path) #  FlowRVS dir

class FlowRVS_SM_Model(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        
        return io.Schema(
            node_id="FlowRVS_SM_Model",
            display_name="FlowRVS_SM_Model",
            category="FlowRVS_SM",
            inputs=[
                io.Combo.Input("wan_dit",options= ["none"] +folder_paths.get_filename_list("diffusion_models") ),
                io.Combo.Input("rvs_dit",options= ["none"] +folder_paths.get_filename_list("FlowRVS") ),       
            ],
            outputs=[
                io.Model.Output(display_name="model"),
                ],
            )
    @classmethod
    def execute(cls,wan_dit, rvs_dit,) -> io.NodeOutput:
        clear_comfyui_cache()
        assert rvs_dit != "none" and wan_dit != "none" ,"need FlowRVS dit  and wan dit"
        args=OmegaConf.create(FlowRVS_SM_origin_dict)
        args.model_id=os.path.join(node_cr_path, "FlowRVS/util/config/Wan2.1-T2V-1.3B-Diffusers")
        args.resume=folder_paths.get_full_path("FlowRVS", rvs_dit)
        args.origin_weights_path=folder_paths.get_full_path("diffusion_models", wan_dit)
        model=load_dit(args)
        return io.NodeOutput(model)
    
class FlowRVS_SM_VAE(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        
        return io.Schema(
            node_id="FlowRVS_SM_VAE",
            display_name="FlowRVS_SM_VAE",
            category="FlowRVS_SM",
            inputs=[
                io.Combo.Input("wan_vae",options= ["none"] + folder_paths.get_filename_list("vae")), 
                io.Combo.Input("rvs_vae",options= ["none"] + folder_paths.get_filename_list("FlowRVS")),           
            ],
            outputs=[
                io.Vae.Output(display_name="vae"),
                ],
            )
    @classmethod
    def execute(cls, wan_vae,rvs_vae) -> io.NodeOutput:
        clear_comfyui_cache()      
        assert  rvs_vae != "none" and wan_vae!="none","need FlowRVS vae and wan vae"
        model_id=os.path.join(node_cr_path, "FlowRVS/util/config/Wan2.1-T2V-1.3B-Diffusers")
        vae_ckpt=folder_paths.get_full_path("FlowRVS", rvs_vae)
        vae_path=folder_paths.get_full_path("vae", wan_vae)
        vae=load_vae_model(model_id,vae_ckpt,vae_path,device)
        return io.NodeOutput(vae)
    
class FlowRVS_SM_Cond(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="FlowRVS_SM_Cond",
            display_name="FlowRVS_SM_Cond",
            category="FlowRVS_SM",
            inputs=[
                io.Vae.Input("vae"),
                io.Conditioning.Input("positive"),
                io.Image.Input("image"),
                io.Float.Input("value", default=1.0, min=0.0, max=1.0,step=0.01,),
                ],
            outputs=[
                io.Conditioning.Output(display_name="cond"),
                     ],

        )
    @classmethod
    def execute(cls, vae,positive,image,value) -> io.NodeOutput:
        clear_comfyui_cache()
        image,original_info=process_video_tensor(image,value)
        cond=data_processor(positive[0][0] ,vae,image,device,dtype=torch.bfloat16)  
        cond["original_info"] = original_info       
        return io.NodeOutput (cond)


class FlowRVS_SM_Decoder(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="FlowRVS_SM_Decoder",
            display_name="FlowRVS_SM_Decoder",
            category="FlowRVS_SM",
            inputs=[
                io.Conditioning.Input("cond"),
                io.Vae.Input("vae"),
                io.Float.Input("threshold", default=0.5, min=0.1, max=1,step=0.1),
                io.Boolean.Input("morphological", default=True),
                io.Int.Input("kernel_size", default=3, min=1, max=10),
                io.Boolean.Input("connected_components", default=True),
                io.Float.Input("min_area_ratio", default=0.01, min=0.001, max=0.1, step=0.001),
                io.Boolean.Input("gaussian_smoothing", default=True),
                io.Float.Input("sigma", default=1.0, min=0.1, max=5.0, step=0.1),
                io.Int.Input("shrink_pixels", default=0, min=0, max=256, step=1, ),
                io.Combo.Input("shrink_method",options= ["uniform","distance"]),
            ],
            outputs=[
                io.Mask.Output(display_name="mask"),
                ],
        )
    @classmethod
    def execute(cls,cond, vae,threshold,morphological,kernel_size,connected_components,min_area_ratio,gaussian_smoothing,sigma,shrink_pixels,shrink_method) -> io.NodeOutput:
        clear_comfyui_cache()
        mask=decode_latents(vae,cond["latents"],cond["origin_h"], cond["origin_w"],cond["original_len"],
                            threshold,device,morphological,connected_components,gaussian_smoothing,shrink_pixels,kernel_size,min_area_ratio,sigma,shrink_method)
        mask=restore_original_size(mask, cond["original_info"],True)
        del cond
        gc_cleanup()
        return io.NodeOutput(mask)


class FlowRVS_SM_KSampler(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="FlowRVS_SM_KSampler",
            display_name="FlowRVS_SM_KSampler",
            category="FlowRVS_SM",
            inputs=[
                io.Model.Input("model"),
                io.Conditioning.Input("cond"),
                io.Int.Input("steps", default=1, min=1, max=10000),
            ],
            outputs=[
                io.Conditioning.Output(display_name="condition"),
            ],
        ) 
    @classmethod
    def execute(cls, model,cond,steps ) -> io.NodeOutput: 
        clear_comfyui_cache()
        latents=inference_single_video( model,steps, os.path.join(node_cr_path, "FlowRVS/util/config/Wan2.1-T2V-1.3B-Diffusers"), cond["x0_video_latent"], cond["prompt_embeds"],device)
        cond["latents"] = latents
        return io.NodeOutput(cond)

class FlowRVS_SM_Apply_Mask(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="FlowRVS_SM_Apply_Mask",
            display_name="FlowRVS_SM_Apply_Mask",
            category="FlowRVS_SM",
            inputs=[
                io.Mask.Input("mask"),
                io.Image.Input("image"),
                io.Boolean.Input("reverse", default=False),
                io.Int.Input("bg_red", default=255, min=0, max=255, step=1),
                io.Int.Input("bg_green", default=255, min=0, max=255,step=1),
                io.Int.Input("bg_blue", default=255, min=0, max=255,step=1 ),
            ],
            outputs=[
                io.Image.Output(display_name="image"),
            ],
        ) 
    @classmethod
    def execute(cls, mask,image,reverse,bg_red,bg_green,bg_blue ) -> io.NodeOutput: 
        mask = mask.clamp(0, 1)
        mask_expanded = mask.unsqueeze(-1).expand_as(image)
        background_color = torch.tensor([bg_red, bg_green, bg_blue], dtype=image.dtype, device=image.device) / 255.0  # 归一化到0-1
        background_tensor = background_color.view(1, 1, 1, -1).expand_as(image)  
        if not reverse:
            image = image * mask_expanded + background_tensor * (1 - mask_expanded)
        else:
            image = image * (1 - mask_expanded) + background_tensor * mask_expanded

        return io.NodeOutput(image)


class FlowRVS_SM_Extension(ComfyExtension):
    @override
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [
            FlowRVS_SM_Model,
            FlowRVS_SM_VAE,
            FlowRVS_SM_Cond,
            FlowRVS_SM_Decoder,
            FlowRVS_SM_KSampler,
            FlowRVS_SM_Apply_Mask,
        ]


async def comfy_entrypoint() -> FlowRVS_SM_Extension:  # ComfyUI calls this to load your extension and its nodes.
    return FlowRVS_SM_Extension()
