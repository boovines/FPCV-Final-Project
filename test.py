import kagglehub

# Download latest version
path = kagglehub.dataset_download("pengbo00/home-fire-dataset")

print("Path to dataset files:", path)