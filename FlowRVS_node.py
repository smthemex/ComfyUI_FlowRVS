 # !/usr/bin/env python
# -*- coding: UTF-8 -*-

import numpy as np
import torch
import os
from omegaconf import OmegaConf

from .model_loader_utils import  FlowRVS_SM_origin_dict,tensor_upscale
from .FlowRVS.inference_demo import inference_single_video,prepare_models,data_processor,process_video_tensor,decode_latents,restore_original_size

import folder_paths
from typing_extensions import override
from comfy_api.latest import ComfyExtension, io
import nodes
import comfy.model_management as mm


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
                io.Combo.Input("rvs_dit",options= ["none"] +folder_paths.get_filename_list("FlowRVS") ),
                io.Combo.Input("rvs_vae",options= ["none"] + folder_paths.get_filename_list("FlowRVS")),
                io.Combo.Input("wan_vae",options= ["none"] + folder_paths.get_filename_list("vae")),            
            ],
            outputs=[
                io.Custom("FlowRVS_DIT").Output("model"),
                io.Custom("FlowRVS_VAE").Output("vae"),
                ],
            )
    @classmethod
    def execute(cls, rvs_dit,rvs_vae,wan_vae,) -> io.NodeOutput:
        
        assert rvs_dit != "none" and rvs_vae != "none" and wan_vae!="none","need FlowRVS dit ,vae and wan diff vae"
        args=OmegaConf.create(FlowRVS_SM_origin_dict)
        args.model_id=os.path.join(node_cr_path, "FlowRVS/util/config/Wan2.1-T2V-1.3B-Diffusers")
        args.resume=folder_paths.get_full_path("FlowRVS", rvs_dit)
        args.vae_ckpt=folder_paths.get_full_path("FlowRVS", rvs_vae)
        args.vae_path=folder_paths.get_full_path("vae", wan_vae)
        model,vae=prepare_models(args)

        return io.NodeOutput(model,vae)
    

class FlowRVS_SM_Cond(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="FlowRVS_SM_Cond",
            display_name="FlowRVS_SM_Cond",
            category="FlowRVS_SM",
            inputs=[
                io.Custom("FlowRVS_VAE").Input("vae"),
                io.Conditioning.Input("condition"),
                io.Image.Input("image"),
                io.Float.Input("value", default=1.0, min=0.0, max=1.0,step=0.01,),
                ],
            outputs=[
                io.Custom("FlowRVS_Con").Output("cond"),
                io.Custom("FlowRVS_Par").Output("param"),
                     ],

        )
    @classmethod
    def execute(cls, vae,condition,image,value) -> io.NodeOutput:
        #text_processor=load_text_processor(os.path.join(node_cr_path, "FlowRVS/util/config/Wan2.1-T2V-1.3B-Diffusers"),clip,device)
        cf_models=mm.loaded_models()
        for model in cf_models:   
            model.unpatch_model(device_to=torch.device("cpu"))
        mm.soft_empty_cache()
        image,original_info=process_video_tensor(image,value)
        cond,param=data_processor(condition[0][0] ,vae,image,device,dtype=torch.bfloat16)  
        param.update({"original_info":original_info})
        return io.NodeOutput (cond,param)


class FlowRVS_SM_Decoder(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="FlowRVS_SM_Encode",
            display_name="FlowRVS_SM_Encode",
            category="FlowRVS_SM",
            inputs=[
                io.Conditioning.Input("condition"),
                io.Custom("FlowRVS_VAE").Input("vae"),
                io.Custom("FlowRVS_Par").Input("param"),
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
    def execute(cls,condition, vae, param,threshold,morphological,kernel_size,connected_components,min_area_ratio,gaussian_smoothing,sigma,shrink_pixels,shrink_method) -> io.NodeOutput:
        mask=decode_latents(vae,condition,param["origin_h"], param["origin_w"],param["original_len"],
                            threshold,device,morphological,connected_components,gaussian_smoothing,shrink_pixels,kernel_size,min_area_ratio,sigma,shrink_method)
        mm.soft_empty_cache()
        mask=restore_original_size(mask, param["original_info"],True)
        return io.NodeOutput(mask)


class FlowRVS_SM_KSampler(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="FlowRVS_SM_KSampler",
            display_name="FlowRVS_SM_KSampler",
            category="FlowRVS_SM",
            inputs=[
                io.Custom("FlowRVS_DIT").Input("model"),
                io.Custom("FlowRVS_Con").Input("cond"),
                io.Int.Input("steps", default=1, min=1, max=10000),
            ],
            outputs=[
                io.Conditioning.Output(display_name="condition"),
            ],
        ) 
    @classmethod
    def execute(cls, model,cond,steps ) -> io.NodeOutput: 
        condition=inference_single_video( model,steps, os.path.join(node_cr_path, "FlowRVS/util/config/Wan2.1-T2V-1.3B-Diffusers"), cond["x0_video_latent"], cond["prompt_embeds"],device)
        return io.NodeOutput(condition)

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


from aiohttp import web
from server import PromptServer
@PromptServer.instance.routes.get("/FlowRVS_SM_Extension")
async def get_hello(request):
    return web.json_response("FlowRVS_SM_Extension")

class FlowRVS_SM_Extension(ComfyExtension):
    @override
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [
            FlowRVS_SM_Model,
            FlowRVS_SM_Cond,
            FlowRVS_SM_Decoder,
            FlowRVS_SM_KSampler,
            FlowRVS_SM_Apply_Mask,
        ]


async def comfy_entrypoint() -> FlowRVS_SM_Extension:  # ComfyUI calls this to load your extension and its nodes.
    return FlowRVS_SM_Extension()
