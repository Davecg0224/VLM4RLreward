import torch
import clip
import cv2
from PIL import Image
import time 

class CLIP():
    def __init__(self, model_version = "ViT-L/14", actionText = ["walking"]):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model, self.preprocess = clip.load(model_version, device=self.device)
        self.text = clip.tokenize(actionText).to(self.device)

    def CLIPprocess(self, image):
        with torch.no_grad():
            logits_per_image, logits_per_text = self.model(image, self.text)
            # [-1, 1] scaled to [0, 1]
            score = logits_per_image[0][0].cpu().numpy()/100
            scaledScore = (score + 1) / 2
        return scaledScore

    def getScore(self, img):
        img_pil = Image.fromarray(img)
        image = self.preprocess(img_pil).unsqueeze(0).to(self.device)
        return self.CLIPprocess(image)

if __name__ == "__main__":

    # select VLM and model version
    # CLIP: ViT-B/16, ViT-L/14, ViT-L/14@336px
    vlm = CLIP(model_version = "ViT-L/14", actionText = ["walking"])
    cap = cv2.VideoCapture('./videos/walking_human.mp4')
    totalScore = 0
    i = 0
    stTime = time.time()
    while cap.isOpened():
        success, img = cap.read()
        if not success:
            # print("Ignoring empty camera frame.")
            # If loading a video, use 'break' instead of 'continue'.
            break

        totalScore += vlm.getScore(img)
        i += 1

    avgScore = totalScore / i
    print("===================================")
    print("Average simliarity score: {:.4f}".format(avgScore))
    endTime = time.time()
    print("FPS: {:.2f}".format(i / (endTime - stTime)))
    print("===================================")