from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="meta-llama/Llama-3.2-1B-Instruct",
    use_auth_token="", # Add your Hugging Face token here
    # Change local_dir to your desired installation path
    local_dir="/home/gregdeli/greg_llms/models/Llama-3.2-1B-Instruct",
)
