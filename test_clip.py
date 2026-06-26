from PIL import Image
import torchvision.transforms as transforms

from core.vlm_tokens.clip_features import CLIPFeatureExtractor

# Load test image
image = Image.open(
    r"D:\hazy-vlm-optical-flow\datasets\kitti\training\image_2\000000_10.png"
).convert("RGB")

# Convert image to tensor
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

image = transform(image)

# Convert to RAFT range [-1, 1]
image = image * 2 - 1

# Add batch dimension
image = image.unsqueeze(0)

# Load CLIP
clip = CLIPFeatureExtractor()

# Extract features
features = clip(image)

print("Feature Shape:", features.shape)
print("\nFirst 10 values:")
print(features[0][:10])