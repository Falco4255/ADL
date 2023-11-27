from PIL import Image, ImageEnhance
import random
import torchvision.transforms as transforms

#Augmentations

#Base inspired from source:
#https://github.com/Spijkervet/SimCLR/blob/master/simclr/modules/transformations/simclr.py
class MaxBrightness(object):
    def __call__(self, img):
        enhancer = ImageEnhance.Brightness(img)
        img = enhancer.enhance(2.0)  # Set brightness to its maximum (2.0)
        return img

class MaxContrast(object):
    def __call__(self, img):
        enhancer = ImageEnhance.Contrast(img)
        img = enhancer.enhance(2.0)  # Set contrast to its maximum (2.0)
        return img

class MaxSaturation(object):
    def __call__(self, img):
        enhancer = ImageEnhance.Color(img)
        img = enhancer.enhance(2.0)  # Set saturation to its maximum (2.0)
        return img

class MaxHue(object):
    def __call__(self, img):
        img = img.convert("HSV")
        hue = random.uniform(0, 360)
        img = img.point(lambda i: i + hue)
        img = img.convert("RGB")
        return img

class RandomApply(object):
    def __init__(self, transform, p=0.5):
        self.transform = transform
        self.p = p

    def __call__(self, img):
        if random.random() < self.p:
            img = self.transform(img)
        return img



class CustomTransforms:
    """
    A stochastic data augmentation module that transforms any given data example randomly
    resulting in two correlated views of the same example,
    denoted x i and x j, which we consider as a positive pair.
    """

    def __init__(self, is_pretrain=True, is_val=False, is_classification = False, is_dino=False):
        self.is_pretrain=is_pretrain
        self.is_val=is_val
        self.is_classification = is_classification
        self.is_dino = is_dino
        s = 1
        color_jitter = transforms.ColorJitter(
            0.8 * s, 0.8 * s, 0.8 * s, 0.2 * s
        )
        self.train_transform1 = transforms.Compose(
            [
                transforms.RandomResizedCrop(size=(128, 128), scale=(0.08, 1.0), ratio=(0.75, 1.333)),
                transforms.RandomHorizontalFlip(),  # with 0.5 probability
                transforms.RandomApply([color_jitter], p=0.8),
                RandomApply(MaxBrightness(), p=0.4),
                RandomApply(MaxContrast(), p=0.4),
                RandomApply(MaxSaturation(), p=0.2),
                RandomApply(MaxHue(), p=0.1),
                transforms.RandomGrayscale(p=0.2),
                transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),

                transforms.ToTensor(),
            ]
        )
        self.train_transform2 = transforms.Compose(
            [
                transforms.RandomResizedCrop(size=(128, 128), scale=(0.08, 1.0), ratio=(0.75, 1.333)),
                transforms.RandomHorizontalFlip(),  # with 0.5 probability
                transforms.RandomApply([color_jitter], p=0.8),
                RandomApply(MaxBrightness(), p=0.4),
                RandomApply(MaxContrast(), p=0.4),
                RandomApply(MaxSaturation(), p=0.2),
                RandomApply(MaxHue(), p=0.1),
                transforms.RandomGrayscale(p=0.2),
                transforms.RandomApply([transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0))], p=0.1),
                transforms.RandomSolarize(5, p=0.2),  # Add solarization transform

                transforms.ToTensor(),
            ]
        )

        self.dino_transform1 = transforms.Compose(
            [
                transforms.RandomResizedCrop(size=(128, 128), scale=(0.5, 1.0), ratio=(0.75, 1.333)),
                transforms.RandomHorizontalFlip(),  # with 0.5 probability
                transforms.RandomApply([color_jitter], p=0.8),
                RandomApply(MaxBrightness(), p=0.4),
                RandomApply(MaxContrast(), p=0.4),
                RandomApply(MaxSaturation(), p=0.2),
                RandomApply(MaxHue(), p=0.1),
                transforms.RandomGrayscale(p=0.2),
                transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),

                transforms.ToTensor(),
            ]
        )

        self.dino_transform2 = transforms.Compose(
            [
                transforms.RandomResizedCrop(size=(128, 128), scale=(0.08, 0.33), ratio=(0.75, 1.333)),
                transforms.RandomHorizontalFlip(),  # with 0.5 probability
                transforms.RandomApply([color_jitter], p=0.8),
                RandomApply(MaxBrightness(), p=0.4),
                RandomApply(MaxContrast(), p=0.4),
                RandomApply(MaxSaturation(), p=0.2),
                RandomApply(MaxHue(), p=0.1),
                transforms.RandomGrayscale(p=0.2),
                transforms.RandomApply([transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0))], p=0.1),
                transforms.RandomSolarize(5, p=0.2),  # Add solarization transform

                transforms.ToTensor(),
            ]
        )

        self.test_transform = transforms.Compose(
            [
                transforms.Resize((128, 128)),
                transforms.ToTensor(),
            ]
        )

        self.classification_transform = transforms.Compose(
            [
                transforms.RandomResizedCrop(size=(128, 128), scale=(0.5, 1.0), ratio=(0.75, 1.333)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
            ]
        )

    def __call__(self, x):
        if self.is_pretrain:
          return self.train_transform1(x), self.train_transform2(x)
        elif self.is_val:
          return self.test_transform(x)
        elif self.is_classification:
          return self.classification_transform(x)
        elif self.is_dino:
            return [self.dino_transform1(x),self.dino_transform1(x),
                    self.dino_transform2(x),self.dino_transform2(x),
                    self.dino_transform2(x),self.dino_transform2(x)]