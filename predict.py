import os
import torch
import traceback
import einops
import numpy as np
import tempfile
from PIL import Image
from cog import BasePredictor, Input, Path

# Assuming diffusers_helper is in the same directory or installed
try:
    from diffusers_helper.hunyuan import encode_prompt_conds, vae_decode, vae_encode, vae_decode_fake
    from diffusers_helper.utils import save_bcthw_as_mp4, crop_or_pad_yield_mask, soft_append_bcthw, resize_and_center_crop
    from diffusers_helper.models.hunyuan_video_packed import HunyuanVideoTransformer3DModelPacked
    from diffusers_helper.pipelines.k_diffusion_hunyuan import sample_hunyuan
    from diffusers_helper.clip_vision import hf_clip_vision_encode
    from diffusers_helper.bucket_tools import find_nearest_bucket
except ImportError:
    print("Make sure the diffusers_helper directory is in the same directory as predict.py")
    raise

from diffusers import AutoencoderKLHunyuanVideo
from transformers import LlamaModel, CLIPTextModel, LlamaTokenizerFast, CLIPTokenizer, SiglipImageProcessor, SiglipVisionModel

# Define the device
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# Cog handles tmp files better, but let's keep this for structure if needed internally
OUTPUTS_DIR = "outputs"
os.makedirs(OUTPUTS_DIR, exist_ok=True)


class Predictor(BasePredictor):
    def setup(self):
        """Load the model into memory to make running multiple predictions efficient"""
        print("Loading models...")
        dtype_transformer = torch.bfloat16
        dtype_others = torch.float16

        MODEL_DIR = "models"

        self.text_encoder = LlamaModel.from_pretrained(
            os.path.join(MODEL_DIR, "text_encoder"), torch_dtype=dtype_others).to(DEVICE)
        self.text_encoder_2 = CLIPTextModel.from_pretrained(
            os.path.join(MODEL_DIR, "text_encoder_2"), torch_dtype=dtype_others).to(DEVICE)
        self.tokenizer = LlamaTokenizerFast.from_pretrained(
            os.path.join(MODEL_DIR, "tokenizer"))
        self.tokenizer_2 = CLIPTokenizer.from_pretrained(
            os.path.join(MODEL_DIR, "tokenizer_2"))
        self.vae = AutoencoderKLHunyuanVideo.from_pretrained(
            os.path.join(MODEL_DIR, "vae"), torch_dtype=dtype_others).to(DEVICE)

        self.feature_extractor = SiglipImageProcessor.from_pretrained(
            os.path.join(MODEL_DIR, "feature_extractor"))
        self.image_encoder = SiglipVisionModel.from_pretrained(
            os.path.join(MODEL_DIR, "image_encoder"), torch_dtype=dtype_others).to(DEVICE)

        self.transformer = HunyuanVideoTransformer3DModelPacked.from_pretrained(
            os.path.join(MODEL_DIR, "transformer"), torch_dtype=dtype_transformer).to(DEVICE)

        self.vae.eval()
        self.text_encoder.eval()
        self.text_encoder_2.eval()
        self.image_encoder.eval()
        self.transformer.eval()

        # Slicing/Tiling might be beneficial even on Replicate GPUs depending on workload
        # self.vae.enable_slicing()
        # self.vae.enable_tiling()

        self.transformer.high_quality_fp32_output_for_inference = True
        print('transformer.high_quality_fp32_output_for_inference = True')

        self.vae.requires_grad_(False)
        self.text_encoder.requires_grad_(False)
        self.text_encoder_2.requires_grad_(False)
        self.image_encoder.requires_grad_(False)
        self.transformer.requires_grad_(False)
        print("Models loaded.")

    @torch.no_grad()
    def predict(
        self,
        input_image: Path = Input(description="Input image"),
        prompt: str = Input(description="Prompt for the video generation"),
        negative_prompt: str = Input(
            description="Negative prompt (currently unused)", default=""),
        seed: int = Input(description="Random seed", default=31337),
        total_second_length: float = Input(
            description="Total Video Length (Seconds)", default=5.0, ge=1, le=120),
        steps: int = Input(
            description="Number of diffusion steps (changing not recommended)", default=25, ge=1, le=100),
        distilled_guidance_scale: float = Input(
            description="Distilled CFG Scale (changing not recommended)", default=10.0, ge=1.0, le=32.0),
        real_guidance_scale: float = Input(
            description="Real CFG Scale (1.0 ignores negative prompt, higher values adhere more strongly to prompt)", default=1.0, ge=1.0, le=32.0),
        guidance_rescale: float = Input(
            description="CFG Rescale (0.0 recommended)", default=0.0, ge=0.0, le=1.0),
        latent_window_size: int = Input(
            description="Latent Window Size (9 recommended)", default=9, ge=1, le=33),
        use_teacache: bool = Input(
            description="Use TeaCache (Faster, may slightly affect hands/fingers)", default=True),
        mp4_crf: int = Input(
            description="MP4 Compression (Lower is better quality, 0=lossless, 16 is good)", default=16, ge=0, le=100)
    ) -> Path:
        """Run a single prediction on the model"""

        # --- Parameters from Gradio that are now fixed or unused ---
        # latent_window_size = 9  # Fixed value from Gradio hidden slider - NOW INPUT
        # # Fixed value from Gradio hidden slider (real_guidance_scale)
        # cfg = 1.0 # NOW INPUT as real_guidance_scale
        # # Fixed value from Gradio hidden slider (guidance_rescale)
        # rs = 0.0 # NOW INPUT as guidance_rescale
        # n_prompt = negative_prompt  # Use the input negative_prompt

        # --- Start of adapted worker logic ---
        total_latent_sections = (
            total_second_length * 30) / (latent_window_size * 4)
        total_latent_sections = int(max(round(total_latent_sections), 1))

        print("Encoding text...")
        llama_vec, clip_l_pooler = encode_prompt_conds(
            prompt, self.text_encoder, self.text_encoder_2, self.tokenizer, self.tokenizer_2)

        # Use negative prompt only if cfg > 1.0
        if real_guidance_scale <= 1.0:
            llama_vec_n = torch.zeros_like(llama_vec)
            clip_l_pooler_n = torch.zeros_like(clip_l_pooler)
            print("Using zero vectors for negative prompt (real_guidance_scale <= 1.0)")
        else:
            print("Encoding negative prompt...")
            # Use the provided negative_prompt
            llama_vec_n, clip_l_pooler_n = encode_prompt_conds(
                negative_prompt, self.text_encoder, self.text_encoder_2, self.tokenizer, self.tokenizer_2)

        llama_vec, llama_attention_mask = crop_or_pad_yield_mask(
            llama_vec, length=512)
        llama_vec_n, llama_attention_mask_n = crop_or_pad_yield_mask(
            llama_vec_n, length=512)
        print("Text encoded.")

        print("Processing input image...")
        # Load image from cog.Path
        input_image_pil = Image.open(input_image)
        input_image_np_orig = np.array(input_image_pil)

        H, W, C = input_image_np_orig.shape
        # Assuming find_nearest_bucket is available
        height, width = find_nearest_bucket(H, W, resolution=640)
        # Assuming resize_and_center_crop is available
        input_image_np = resize_and_center_crop(
            input_image_np_orig, target_width=width, target_height=height)

        input_image_pt = torch.from_numpy(
            input_image_np).float().to(DEVICE) / 127.5 - 1.0
        input_image_pt = input_image_pt.permute(
            2, 0, 1)[None, :, None]  # B C T H W
        print("Input image processed.")

        print("VAE encoding...")
        start_latent = vae_encode(input_image_pt, self.vae)
        print("VAE encoded.")

        print("CLIP Vision encoding...")
        image_encoder_output = hf_clip_vision_encode(
            input_image_np, self.feature_extractor, self.image_encoder)
        image_encoder_last_hidden_state = image_encoder_output.last_hidden_state
        print("CLIP Vision encoded.")

        # Dtype adjustments
        llama_vec = llama_vec.to(self.transformer.dtype)
        llama_vec_n = llama_vec_n.to(self.transformer.dtype)
        clip_l_pooler = clip_l_pooler.to(self.transformer.dtype)
        clip_l_pooler_n = clip_l_pooler_n.to(self.transformer.dtype)
        image_encoder_last_hidden_state = image_encoder_last_hidden_state.to(
            self.transformer.dtype)

        print("Starting sampling...")
        rnd = torch.Generator(DEVICE).manual_seed(seed)
        num_frames = latent_window_size * 4 - 3  # e.g. 9*4-3 = 33

        # Initialize history buffers on CPU to save GPU memory if needed, move later
        history_latents = torch.zeros(
            size=(1, 16, 1 + 2 + 16, height // 8, width // 8), dtype=torch.float32).cpu()
        history_pixels = None
        total_generated_latent_frames = 0
        output_filename = None  # Track the last generated file

        latent_paddings = reversed(range(total_latent_sections))

        if total_latent_sections > 4:
            # Same padding trick as in the original script
            latent_paddings = [3] + [2] * (total_latent_sections - 3) + [1, 0]

        for i, latent_padding in enumerate(latent_paddings):
            is_last_section = latent_padding == 0
            latent_padding_size = latent_padding * latent_window_size

            print(
                f'Section {i+1}/{total_latent_sections}: latent_padding_size = {latent_padding_size}, is_last_section = {is_last_section}')

            # Calculate indices for sampling
            indices = torch.arange(0, sum(
                [1, latent_padding_size, latent_window_size, 1, 2, 16])).unsqueeze(0).to(DEVICE)
            clean_latent_indices_pre, blank_indices, latent_indices, clean_latent_indices_post, clean_latent_2x_indices, clean_latent_4x_indices = indices.split(
                [1, latent_padding_size, latent_window_size, 1, 2, 16], dim=1)
            clean_latent_indices = torch.cat(
                [clean_latent_indices_pre, clean_latent_indices_post], dim=1)

            # Prepare clean latents from history and start latent
            # Move start_latent to device where history is
            clean_latents_pre = start_latent.to(
                history_latents.device, dtype=history_latents.dtype)
            # Ensure history_latents is on the correct device before slicing
            current_history_latents_device = history_latents.to(DEVICE)
            clean_latents_post, clean_latents_2x, clean_latents_4x = current_history_latents_device[:, :, :1 + 2 + 16, :, :].split([
                                                                                                                                   1, 2, 16], dim=2)
            clean_latents = torch.cat(
                [clean_latents_pre.to(DEVICE), clean_latents_post], dim=2)

            # TeaCache setup
            if use_teacache:
                self.transformer.initialize_teacache(
                    enable_teacache=True, num_steps=steps)
            else:
                self.transformer.initialize_teacache(enable_teacache=False)

            # Replicate doesn't have interactive callbacks in the same way Gradio does.
            # We can print progress or potentially yield intermediate results if Replicate supports it.
            # For now, just removing the callback logic.
            def callback_stub(d):
                current_step = d['i'] + 1
                print(f"  Sampling step {current_step}/{steps}")
                # Could potentially yield progress here if needed in the future
                # yield f"Sampling step {current_step}/{steps}"
                pass

            generated_latents = sample_hunyuan(
                transformer=self.transformer,
                sampler='unipc',  # Or allow as input? Gradio used unipc
                width=width,
                height=height,
                frames=num_frames,
                real_guidance_scale=real_guidance_scale,
                distilled_guidance_scale=distilled_guidance_scale,
                guidance_rescale=guidance_rescale,
                num_inference_steps=steps,
                generator=rnd,
                prompt_embeds=llama_vec.to(DEVICE),
                prompt_embeds_mask=llama_attention_mask.to(DEVICE),
                prompt_poolers=clip_l_pooler.to(DEVICE),
                negative_prompt_embeds=llama_vec_n.to(DEVICE),
                negative_prompt_embeds_mask=llama_attention_mask_n.to(DEVICE),
                negative_prompt_poolers=clip_l_pooler_n.to(DEVICE),
                device=DEVICE,
                dtype=self.transformer.dtype,
                image_embeddings=image_encoder_last_hidden_state.to(DEVICE),
                latent_indices=latent_indices.to(DEVICE),
                clean_latents=clean_latents.to(DEVICE),
                clean_latent_indices=clean_latent_indices.to(DEVICE),
                clean_latents_2x=clean_latents_2x.to(DEVICE),
                clean_latent_2x_indices=clean_latent_2x_indices.to(DEVICE),
                clean_latents_4x=clean_latents_4x.to(DEVICE),
                clean_latent_4x_indices=clean_latent_4x_indices.to(DEVICE),
                callback=callback_stub,  # Use stub or remove if no progress needed
            )

            if is_last_section:
                # Prepend the start latent for the final section
                generated_latents = torch.cat(
                    [start_latent.to(generated_latents.device), generated_latents], dim=2)

            total_generated_latent_frames += int(generated_latents.shape[2])
            print(
                f"  Generated latents shape: {generated_latents.shape}, total frames: {total_generated_latent_frames}")

            # Append new latents to history (ensure devices match, keep history on CPU if large)
            history_latents = torch.cat(
                [generated_latents.cpu(), history_latents], dim=2)

            # Decode the current full history for output
            print("  VAE decoding...")
            real_history_latents = history_latents[:, :, :total_generated_latent_frames, :, :].to(
                DEVICE)  # Move relevant part to GPU for VAE

            # Simpler decoding for now: decode the whole sequence each time.
            # The soft_append logic might be complex to replicate exactly without Gradio's streaming.
            # This might be less efficient but ensures correctness for the final output.
            # If memory becomes an issue, we might need to decode sections and stitch.
            # Decode on GPU, move result to CPU
            decoded_pixels = vae_decode(real_history_latents, self.vae).cpu()

            # Save the intermediate/final video
            # Use a temporary file for the output
            out_path = Path(tempfile.mkdtemp()) / "output.mp4"
            output_filename = str(out_path)

            print(f"  Saving video to {output_filename}...")
            save_bcthw_as_mp4(decoded_pixels, output_filename,
                              fps=30, crf=mp4_crf)
            print(f"  Video saved. Pixel shape: {decoded_pixels.shape}")

            # Update history_pixels (used in original soft_append, maybe remove if not needed)
            history_pixels = decoded_pixels  # Store the latest full decode

            if is_last_section:
                print("Final section processed.")
                break

        print("Sampling finished.")

        if output_filename is None:
            raise RuntimeError("No video file was generated.")

        # Return the path to the final generated video file
        return Path(output_filename)
