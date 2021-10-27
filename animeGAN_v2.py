import os
import cv2
import torch
from networks import Generator


class AnimeGAN_v2:
    def __init__(self, **kwargs: dict) -> None:
        """ Architecture of style transfer network """
        super(AnimeGAN_v2, self).__init__()
        self.model_name: str = 'animeGAN_v2'
        self.model_version: str = '1.0.0'
            
        self.pretrained_model_path: str = kwargs['pretrained_model_path']

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        torch.set_grad_enabled(False)

        self.model = Generator().eval().to(self.device)
        ckpt = torch.load(self.pretrained_model_path, map_location=self.device)
        self.model.load_state_dict(ckpt)

    def generate_new_image(self, source_image):
        content_image = self.image2tensor(source_image)
        output = self.model(content_image.to(self.device))
        output_array = output[0].permute(1, 2, 0).detach().cpu().numpy()
        return (0.5 * output_array + 0.5).clip(0, 1)

    @staticmethod
    def image2tensor(image):
        image = torch.FloatTensor(image).permute(2, 0, 1).unsqueeze(0) / 255.
        return (image - 0.5) / 0.5

    @staticmethod
    def tensor2image(tensor):
        tensor = tensor.clamp_(-1., 1.).detach().squeeze().permute(1, 2, 0).cpu().numpy()
        return tensor * 0.5 + 0.5
