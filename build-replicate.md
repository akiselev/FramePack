
# Tutorial: Building a Docker Image for FramePack Deployment on Replicate

This tutorial will guide you through building a Docker image for your FramePack model deployment on Replicate, using a LambdaLabs cloud instance for faster downloading and building.

## Part 1: Setting Up LambdaLabs Instance

1. Launch a LambdaLabs instance with sufficient storage (at least 100GB recommended)
2. SSH into your instance:
   ```bash
   ssh ubuntu@<your-lambda-instance-ip>
   ```

## Part 2: Setting Up the Project

1. Create a project directory and clone your FramePack repository:
   ```bash
   mkdir -p ~/replicate-build
   cd ~/replicate-build
   git clone <your-framepack-repo-url> .
   # or copy your files using scp if they're not in a git repo
   ```

2. Install Cog (Replicate's packaging tool):
   ```bash
   sudo curl -o /usr/local/bin/cog -L https://github.com/replicate/cog/releases/latest/download/cog_`uname -s`_`uname -m`
   sudo chmod +x /usr/local/bin/cog
   ```

## Part 3: Download Models Ahead of Time

Create a script to download all models beforehand:

```bash
nano download_models.py
```

Paste this content:

```python
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
LlamaModel.from_pretrained("hunyuanvideo-community/HunyuanVideo", subfolder='text_encoder').save_pretrained(model_paths["text_encoder"])

print("Downloading text_encoder_2...")
CLIPTextModel.from_pretrained("hunyuanvideo-community/HunyuanVideo", subfolder='text_encoder_2').save_pretrained(model_paths["text_encoder_2"])

print("Downloading tokenizer...")
LlamaTokenizerFast.from_pretrained("hunyuanvideo-community/HunyuanVideo", subfolder='tokenizer').save_pretrained(model_paths["tokenizer"])

print("Downloading tokenizer_2...")
CLIPTokenizer.from_pretrained("hunyuanvideo-community/HunyuanVideo", subfolder='tokenizer_2').save_pretrained(model_paths["tokenizer_2"])

print("Downloading VAE...")
AutoencoderKLHunyuanVideo.from_pretrained("hunyuanvideo-community/HunyuanVideo", subfolder='vae').save_pretrained(model_paths["vae"])

print("Downloading feature_extractor...")
SiglipImageProcessor.from_pretrained("lllyasviel/flux_redux_bfl", subfolder='feature_extractor').save_pretrained(model_paths["feature_extractor"])

print("Downloading image_encoder...")
SiglipVisionModel.from_pretrained("lllyasviel/flux_redux_bfl", subfolder='image_encoder').save_pretrained(model_paths["image_encoder"])

print("Downloading transformer...")
HunyuanVideoTransformer3DModelPacked.from_pretrained('lllyasviel/FramePackI2V_HY').save_pretrained(model_paths["transformer"])

print("All models downloaded successfully.")
```

Install the required packages and run the script:

```bash
pip install diffusers transformers safetensors accelerate
# Install any other dependencies your diffusers_helper module requires
python download_models.py
```

## Part 4: Modify predict.py to Load Local Models

Edit `predict.py` to load models from local paths:

```bash
nano predict.py
```

Make these changes in the `setup` method:

```python
def setup(self):
    """Load the model into memory to make running multiple predictions efficient"""
    print("Loading models from local paths...")
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
    
    # Rest of the setup code remains the same
    self.vae.eval()
    # ...
```

## Part 5: Create cog.yaml Configuration

Create the Cog configuration file:

```bash
nano cog.yaml
```

Add this content:

```yaml
build:
  gpu: true
  cuda: "11.8"
  python_version: "3.10"
  python_packages:
    - torch==2.0.1
    - diffusers==0.21.4
    - transformers==4.35.2
    - safetensors==0.4.0
    - numpy==1.24.3
    - einops==0.6.1
    - pillow==10.0.0
    - moviepy==1.0.3
    # Add any other dependencies needed by your diffusers_helper module

predict: "predict.py:Predictor"
```

## Part 6: Create a Dockerfile

```bash
nano Dockerfile
```

Add this content:

```dockerfile
FROM r8.im/cog-nvidia-cuda:11.8

# Install Python and pip
RUN apt-get update && apt-get install -y python3 python3-pip ffmpeg

# Install Cog
RUN curl -o /usr/local/bin/cog -L https://github.com/replicate/cog/releases/latest/download/cog_`uname -s`_`uname -m` && \
    chmod +x /usr/local/bin/cog

# Set working directory
WORKDIR /app

# Copy requirements
COPY cog.yaml /app/

# Install Python dependencies
RUN python3 -m pip install --upgrade pip && \
    cog install-requirements

# Copy source code and models
COPY . /app/
COPY models /app/models
COPY diffusers_helper /app/diffusers_helper
COPY predict.py /app/

# Set default command
ENTRYPOINT ["cog", "predict"]
```

## Part 7: Build and Push to Replicate

1. Build the Docker image:
   ```bash
   cog build
   ```

2. Test the model locally:
   ```bash
   cog predict -i input_image=@path/to/test_image.jpg -i prompt="A girl dancing gracefully"
   ```

3. Log in to Replicate:
   ```bash
   cog login
   ```

4. Push to Replicate:
   ```bash
   cog push r8.im/yourusername/framepack
   ```

## Additional Tips and Troubleshooting

1. **Large Model Files**: If your model files are very large:
   ```bash
   # Check the size of models directory
   du -sh models/
   
   # If needed, use git-lfs or split large files
   ```

2. **Memory Issues During Build**: If you encounter memory issues while building:
   ```bash
   # Increase swap space
   sudo fallocate -l 16G /swapfile
   sudo chmod 600 /swapfile
   sudo mkswap /swapfile
   sudo swapon /swapfile
   ```

3. **Network Issues**: If downloads are interrupted:
   ```bash
   # Resume downloads with longer timeout
   TRANSFORMERS_REQUEST_TIMEOUT=600 python download_models.py
   ```

4. **Diffusers Helper Module**: Ensure you have the `diffusers_helper` directory properly included in your build, as it's imported in your `predict.py` script.

5. **Test Before Pushing**: Always test the Docker image locally before pushing to Replicate to ensure it works as expected.

This tutorial should help you successfully build and push your FramePack model to Replicate. The key is downloading the models ahead of time and including them in your Docker image, rather than downloading at runtime.
