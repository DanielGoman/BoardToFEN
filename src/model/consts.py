import torchvision.transforms as transforms

default_image_size = (64, 64)
default_transforms = [transforms.ToTensor(),
                      transforms.Resize(default_image_size),
                      transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                           std=[0.229, 0.224, 0.225])]

