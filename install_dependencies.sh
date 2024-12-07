# install torch [or you can skip this step and let vllm to install the correct version for you]
# pip install torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 --index-url https://download.pytorch.org/whl/cu121

# install vllm
pip3 install vllm==0.5.4
pip3 install ray==2.10 # other version may have bug

# flash attention 2
pip3 install flash-attn --no-build-isolation