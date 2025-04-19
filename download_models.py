import os
from transformers import LlamaModel, CLIPTextModel, LlamaTokenizerFast, CLIPTokenizer, SiglipImageProcessor, SiglipVisionModel
from diffusers import AutoencoderKLHunyuanVideo
from diffusers_helper.models.hunyuan_video_packed import HunyuanVideoTransformer3DModelPacked

# Create the models directory
os.makedirs("models", exist_ok=True)

# Set up paths
model_paths = {
    "text_encoder": "models/text_encoder",
    "text_encoder_2": "models/text_encoder_2",
    "tokenizer": "models/tokenizer",
    "tokenizer_2": "models/tokenizer_2",
    "vae": "models/vae",
    "feature_extractor": "models/feature_extractor",
    "image_encoder": "models/image_encoder",
    "transformer": "models/transformer"
}

print("Downloading text_encoder...")
LlamaModel.from_pretrained("hunyuanvideo-community/HunyuanVideo",
                           subfolder='text_encoder').save_pretrained(model_paths["text_encoder"])

print("Downloading text_encoder_2...")
CLIPTextModel.from_pretrained("hunyuanvideo-community/HunyuanVideo",
                              subfolder='text_encoder_2').save_pretrained(model_paths["text_encoder_2"])

print("Downloading tokenizer...")
LlamaTokenizerFast.from_pretrained("hunyuanvideo-community/HunyuanVideo",
                                   subfolder='tokenizer').save_pretrained(model_paths["tokenizer"])

print("Downloading tokenizer_2...")
CLIPTokenizer.from_pretrained("hunyuanvideo-community/HunyuanVideo",
                              subfolder='tokenizer_2').save_pretrained(model_paths["tokenizer_2"])

print("Downloading VAE...")
AutoencoderKLHunyuanVideo.from_pretrained(
    "hunyuanvideo-community/HunyuanVideo", subfolder='vae').save_pretrained(model_paths["vae"])

print("Downloading feature_extractor...")
SiglipImageProcessor.from_pretrained(
    "lllyasviel/flux_redux_bfl", subfolder='feature_extractor').save_pretrained(model_paths["feature_extractor"])

print("Downloading image_encoder...")
SiglipVisionModel.from_pretrained(
    "lllyasviel/flux_redux_bfl", subfolder='image_encoder').save_pretrained(model_paths["image_encoder"])

print("Downloading transformer...")
HunyuanVideoTransformer3DModelPacked.from_pretrained(
    'lllyasviel/FramePackI2V_HY').save_pretrained(model_paths["transformer"])

print("All models downloaded successfully.")
