# data augmentation for cifar-10

from torchvision import transforms

cifar10_mean = (0.4913, 0.4821, 0.4465)
cifar10_std = (0.2470, 0.2434, 0.2615)

def get_random_resized_crop(size):
    random_resized_crop = transforms.Compose([
        transforms.RandomResizedCrop(size),
        transforms.RandomHorizontalFlip()
    ])
    
    return random_resized_crop
    

def get_color_distortion(s=1.0):
    # code from official paper Appendix A
    # s is the strength of color distortion.
    color_jitter = transforms.ColorJitter(0.8*s, 0.8*s, 0.8*s, 0.2*s)
    rnd_color_jitter = transforms.RandomApply([color_jitter], p=0.8)
    rnd_gray = transforms.RandomGrayscale(p=0.2)
    color_distort = transforms.Compose([
        rnd_color_jitter,
        rnd_gray
        ])
    
    return color_distort


# TODO: implement Gaussian Blur


def get_data_augmentation_for_cifar10(size, s=1.0):
    # according to Appendix B.7 of the original paper, authors did not use Gaussian blur for cifar-10
    
    data_aug = transforms.Compose([
        get_random_resized_crop(size),
        get_color_distortion(s=s),
        transforms.ToTensor(),
        transforms.Normalize(mean=cifar10_mean, std=cifar10_std)
    ])
    
    return data_aug


class TransformTwice:
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, image):
        x_i = self.transform(image)
        x_j = self.transform(image)
        
        return x_i, x_j
        