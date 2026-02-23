# ComfyUI_FlowRVS
[FlowRVS](https://github.com/xmz111/FlowRVS)，Deforming Videos to Masks: Flow Matching for Referring Video Segmentation，you can use it in comfyUI when use this node.

# Update
* support new checkpoints,and fix some bugs
* 支持新的模型，修复一些bug

# 1. Installation
In the ./ComfyUI/custom_nodes directory, run the following:   
```
git clone https://github.com/smthemex/ComfyUI_FlowRVS.git
```

# 2. Requirements
```
pip install -r requirements.txt
```

# 3.Model
* 3.1 FlowRVS dit  and  vae from [here](https://huggingface.co/xmz111/FlowRVS/tree/main)   / 官方底模和vae  
* 3.2 ‘diffuser vae’ and transformer  download [here](https://huggingface.co/Wan-AI/Wan2.1-T2V-1.3B-Diffusers)  / 暂不兼容comfyorg的vae  
* 3.3 T5 download [here](https://huggingface.co/Comfy-Org/Wan_2.1_ComfyUI_repackaged/tree/main/split_files/text_encoders)  /  comfy官方T5 FP8  
```
--  ComfyUI/models
    |-- vae
        |-- diffusion_pytorch_model.safetensors  # rename diffsuers vae   最好重命名
    |-- clip
        |-- umt5_xxl_fp8_e4m3fn_scaled.safetensors  #  or umt5_xxl_fp16.safetensors
    |-- diffusion_models
        |--  diffusion_pytorch_model.safetensors  #   rename diffsuers transformers   最好重命名
    |-- FlowRVS
        |-- FlowRVS_dit_mevis.pth #2.64G # new checkpoints
        |--tuned_vae.pth
```
  

# 4.Example
![](https://github.com/smthemex/ComfyUI_FlowRVS/blob/main/example_workflows/example.png)
![](https://github.com/smthemex/ComfyUI_FlowRVS/blob/main/example_workflows/example.gif)

# 5.Citation

```
@article{wang2025flowrvs,
  title={Deforming Videos to Masks: Flow Matching for Referring Video Segmentation},
  author={Wang, Zanyi and Jiang, Dengyang and Li, Liuzhuozheng and Dang, Sizhe and Li, Chengzu and Yang, Harry and Dai, Guang and Wang, Mengmeng and Wang, Jingdong},
  journal={arXiv preprint arXiv:2510.06139}, 
  year={2025}
}
```
