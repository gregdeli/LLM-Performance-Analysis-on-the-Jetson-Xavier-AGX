from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="Qwen/Qwen1.5-MoE-A2.7B",
    use_auth_token="", # Add your Hugging Face token here
    # Change local_dir to your desired installation path
    local_dir="/home/gregdeli/greg_llms/models/Qwen1.5-MoE-A2.7B",
)
