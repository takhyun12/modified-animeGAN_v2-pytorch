import os
import cv2
import torch
from models.animeGAN_v2.model import Generator
import warnings

warnings.filterwarnings('ignore')


class AnimeGAN_v2:
    def __init__(self, **kwargs: dict) -> None:
        """ Architecture of style transfer network """
        super(AnimeGAN_v2, self).__init__()
        self.model_name: str = 'animeGAN_v2'
        self.model_version: str = '1.0.0'

        self.content_image = kwargs['content_image']
        self.content_video = kwargs['content_video']
        self.video_fx = kwargs['video_fx']
        self.video_fy = kwargs['video_fy']

        self.pretrained_model_path: str = kwargs['pretrained_model_path']
        if len(self.pretrained_model_path) != 0:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            torch.set_grad_enabled(False)

            self.model = Generator().eval().to(self.device)
            ckpt = torch.load(self.pretrained_model_path, map_location=self.device)
            self.model.load_state_dict(ckpt)

    def generate_new_image(self):
        content_image = AnimeGAN_v2.load_image(self.content_image)
        output = self.model(content_image.to(self.device))
        output_array = output[0].permute(1, 2, 0).detach().cpu().numpy()
        return (0.5 * output_array + 0.5).clip(0, 1)

    def generate_new_video(self):
        output_path: str = './output'
        if not os.path.exists(output_path):
            os.mkdir(output_path)
        output_path = f"{output_path}/anime_output.mp4"

        # load video
        video_path = self.content_video

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return

        # down sampling size prediction
        fx, fy = self.video_fx, self.video_fy
        ret, img = cap.read()
        resized_img = cv2.resize(img, dsize=(0, 0), fx=fx, fy=fy, interpolation=cv2.INTER_LINEAR)
        output_height, output_width = resized_img.shape[:2]

        # codec and fps
        fourcc = cv2.VideoWriter_fourcc(*'avc1')
        video_fps = cap.get(cv2.CAP_PROP_FPS)

        output_video = cv2.VideoWriter(f'{output_path}', fourcc, video_fps, (output_width, output_height))
        while cap.isOpened():
            ret, frame = cap.read()
            if ret:
                frame = cv2.resize(frame, dsize=(0, 0), fx=fx, fy=fy, interpolation=cv2.INTER_LINEAR)
                output_frame = self.generate_new_frame(frame)
                output_frame = cv2.convertScaleAbs(output_frame, alpha=255.0)
                output_video.write(output_frame)
            else:
                break

        cap.release()
        output_video.release()
        cv2.destroyAllWindows()

        return output_path

    def generate_new_frame(self, frame):
        content_image = self.image2tensor(frame)
        output = self.model(content_image.to(self.device))
        output_array = output[0].permute(1, 2, 0).detach().cpu().numpy()
        return (0.5 * output_array + 0.5).clip(0, 1)

    @staticmethod
    def load_image(content_image, size=None):
        image = AnimeGAN_v2.image2tensor(content_image)
        return image

    @staticmethod
    def image2tensor(image):
        image = torch.FloatTensor(image).permute(2, 0, 1).unsqueeze(0) / 255.
        return (image - 0.5) / 0.5

    @staticmethod
    def tensor2image(tensor):
        tensor = tensor.clamp_(-1., 1.).detach().squeeze().permute(1, 2, 0).cpu().numpy()
        return tensor * 0.5 + 0.5
