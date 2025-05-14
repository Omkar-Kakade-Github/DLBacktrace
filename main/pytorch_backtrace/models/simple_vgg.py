import torch
import torch.nn as nn
import torchvision # Added for loading pretrained models

class VGG19(nn.Module):
    def __init__(self, num_classes=1000):
        super(VGG19, self).__init__()
        self.identity = nn.Identity() 
        self.conv1_1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.relu1_1 = nn.ReLU(inplace=True)
        self.conv1_2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.relu1_2 = nn.ReLU(inplace=True)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2_1 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.relu2_1 = nn.ReLU(inplace=True)
        self.conv2_2 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.relu2_2 = nn.ReLU(inplace=True)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv3_1 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.relu3_1 = nn.ReLU(inplace=True)
        self.conv3_2 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.relu3_2 = nn.ReLU(inplace=True)
        self.conv3_3 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.relu3_3 = nn.ReLU(inplace=True)
        self.conv3_4 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.relu3_4 = nn.ReLU(inplace=True)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv4_1 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.relu4_1 = nn.ReLU(inplace=True)
        self.conv4_2 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.relu4_2 = nn.ReLU(inplace=True)
        self.conv4_3 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.relu4_3 = nn.ReLU(inplace=True)
        self.conv4_4 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.relu4_4 = nn.ReLU(inplace=True)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv5_1 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.relu5_1 = nn.ReLU(inplace=True)
        self.conv5_2 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.relu5_2 = nn.ReLU(inplace=True)
        self.conv5_3 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.relu5_3 = nn.ReLU(inplace=True)
        self.conv5_4 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.relu5_4 = nn.ReLU(inplace=True)
        self.pool5 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(512 * 7 * 7, 4096)
        self.relu_fc1 = nn.ReLU(inplace=True)
        self.dropout1 = nn.Dropout()
        self.fc2 = nn.Linear(4096, 4096)
        self.relu_fc2 = nn.ReLU(inplace=True)
        self.dropout2 = nn.Dropout()
        self.fc3 = nn.Linear(4096, num_classes)

    def forward(self, x):
        x = self.identity(x)
        x = self.conv1_1(x)
        x = self.relu1_1(x)
        x = self.conv1_2(x)
        x = self.relu1_2(x)
        x = self.pool1(x)
        x = self.conv2_1(x)
        x = self.relu2_1(x)
        x = self.conv2_2(x)
        x = self.relu2_2(x)
        x = self.pool2(x)
        x = self.conv3_1(x)
        x = self.relu3_1(x)
        x = self.conv3_2(x)
        x = self.relu3_2(x)
        x = self.conv3_3(x)
        x = self.relu3_3(x)
        x = self.conv3_4(x)
        x = self.relu3_4(x)
        x = self.pool3(x)
        x = self.conv4_1(x)
        x = self.relu4_1(x)
        x = self.conv4_2(x)
        x = self.relu4_2(x)
        x = self.conv4_3(x)
        x = self.relu4_3(x)
        x = self.conv4_4(x)
        x = self.relu4_4(x)
        x = self.pool4(x)
        x = self.conv5_1(x)
        x = self.relu5_1(x)
        x = self.conv5_2(x)
        x = self.relu5_2(x)
        x = self.conv5_3(x)
        x = self.relu5_3(x)
        x = self.conv5_4(x)
        x = self.relu5_4(x)
        x = self.pool5(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu_fc1(x)
        x = self.dropout1(x)
        x = self.fc2(x)
        x = self.relu_fc2(x)
        x = self.dropout2(x)
        x = self.fc3(x)
        return x

def load_pretrained_vgg(num_classes=200):
    """Loads a pretrained VGG-19 model and maps weights to custom VGG."""
    pretrained_vgg = torchvision.models.vgg19(weights=torchvision.models.VGG19_Weights.IMAGENET1K_V1)
    pretrained_dict = pretrained_vgg.state_dict()
    custom_vgg = VGG19(num_classes=num_classes)
    custom_dict = custom_vgg.state_dict()
    
    # Mapping from torchvision VGG19 feature layer names to custom VGG19 layer names
    # This mapping needs to be comprehensive for all layers you want to transfer.
    # The example only showed a few conv layers, extending it here for more.
    mapping = {
        # Block 1
        "features.0.weight": "conv1_1.weight", "features.0.bias": "conv1_1.bias",
        "features.2.weight": "conv1_2.weight", "features.2.bias": "conv1_2.bias",
        # Block 2
        "features.5.weight": "conv2_1.weight", "features.5.bias": "conv2_1.bias",
        "features.7.weight": "conv2_2.weight", "features.7.bias": "conv2_2.bias",
        # Block 3
        "features.10.weight": "conv3_1.weight", "features.10.bias": "conv3_1.bias",
        "features.12.weight": "conv3_2.weight", "features.12.bias": "conv3_2.bias",
        "features.14.weight": "conv3_3.weight", "features.14.bias": "conv3_3.bias",
        "features.16.weight": "conv3_4.weight", "features.16.bias": "conv3_4.bias",
        # Block 4
        "features.19.weight": "conv4_1.weight", "features.19.bias": "conv4_1.bias",
        "features.21.weight": "conv4_2.weight", "features.21.bias": "conv4_2.bias",
        "features.23.weight": "conv4_3.weight", "features.23.bias": "conv4_3.bias",
        "features.25.weight": "conv4_4.weight", "features.25.bias": "conv4_4.bias",
        # Block 5
        "features.28.weight": "conv5_1.weight", "features.28.bias": "conv5_1.bias",
        "features.30.weight": "conv5_2.weight", "features.30.bias": "conv5_2.bias",
        "features.32.weight": "conv5_3.weight", "features.32.bias": "conv5_3.bias",
        "features.34.weight": "conv5_4.weight", "features.34.bias": "conv5_4.bias",
        # Classifier - Note: torchvision VGG19 classifier layers are 0, 3, 6
        # Your custom VGG19 fc layers are fc1, fc2, fc3
        # Sizes must match or you need to handle resizing/reinitialization.
        # For IMAGENET1K_V1, classifier[0] is Linear(in_features=25088, out_features=4096)
        # Your fc1 is Linear(512 * 7 * 7, 4096) which is Linear(25088, 4096) - MATCHES
        "classifier.0.weight": "fc1.weight", "classifier.0.bias": "fc1.bias",
        "classifier.3.weight": "fc2.weight", "classifier.3.bias": "fc2.bias",
        # Classifier's last layer (classifier.6) maps to num_classes (1000 for ImageNet)
        # Your fc3 maps to `num_classes` (default 1000, or 200 in this function call)
        # If `num_classes` is different from 1000, this layer fc3 should NOT be loaded from pretrained,
        # as it needs to be randomly initialized for fine-tuning on the new number of classes.
    }

    new_pretrained_dict = {}
    for torchvision_name, custom_name in mapping.items():
        if torchvision_name in pretrained_dict and custom_name in custom_dict:
            if pretrained_dict[torchvision_name].shape == custom_dict[custom_name].shape:
                 new_pretrained_dict[custom_name] = pretrained_dict[torchvision_name]
            # else: print(f"Shape mismatch for {custom_name}: Pretrained {pretrained_dict[torchvision_name].shape}, Custom {custom_dict[custom_name].shape}")
        # else: print(f"Key mismatch: {torchvision_name} or {custom_name} not found.")

    custom_dict.update(new_pretrained_dict)
    
    # Load the weights. `strict=False` allows for missing keys (e.g. fc3 if num_classes differs)
    custom_vgg.load_state_dict(custom_dict, strict=False)
    
    # If num_classes is different from the pretrained model's num_classes (1000 for ImageNet),
    # reinitialize the final classification layer (fc3).
    if num_classes != 1000: # Assuming ImageNet pretrained has 1000 classes
        custom_vgg.fc3 = nn.Linear(custom_vgg.fc3.in_features, num_classes)
        # print(f"Reinitialized fc3 for {num_classes} classes.")

    return custom_vgg

if __name__ == '__main__':
    # Example usage:
    # Input for VGG19 is typically 3x224x224
    dummy_input = torch.randn(1, 3, 224, 224) 
    
    # Load custom VGG19 with 10 classes, trying to use pretrained weights
    print("Loading VGG19 with custom number of classes (e.g., 10) and pretrained weights...")
    model = load_pretrained_vgg(num_classes=10) 
    # print(model) # quite long
    
    output = model(dummy_input)
    print("Output shape:", output.shape)

    # Example: Load with default 1000 classes (ImageNet size)
    # print("\nLoading VGG19 with 1000 classes (ImageNet default) and pretrained weights...")
    # model_imagenet = load_pretrained_vgg(num_classes=1000)
    # output_imagenet = model_imagenet(dummy_input)
    # print("Output shape (ImageNet classes):", output_imagenet.shape)

    # To use with Backtrace, you would initialize it:
    # from DLBacktrace.main.pytorch_backtrace.logic.backtrace import BacktracePyTorch (or Backtrace if that's the final class name)
    # backtrace_obj = BacktracePyTorch(model=model) 
    # ... then proceed to get activations and run eval ... 
