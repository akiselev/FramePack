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