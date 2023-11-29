import torch
# import clip
import cv2
from PIL import Image
import time 
from gpt import GPT
import numpy as np

import matplotlib.pyplot as plt
from lavis.models import model_zoo, load_model_and_preprocess

class VLM():
    def __init__(self, model_name = 'clip_feature_extractor',  model_version = "ViT-L-14", actionText = ["walking"]):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model, self.vis_processors, self.txt_processors = load_model_and_preprocess(name = model_name, 
                                                                model_type = model_version, 
                                                                is_eval=True,
                                                                device=self.device)
        # self.text = clip.tokenize(actionText).to(self.device)
        self.text = [self.txt_processors["eval"](f"a human is {c}") for c in actionText]

    # def CLIPprocess(self, image):
    #     with torch.no_grad():
    #         logits_per_image, logits_per_text = self.model(image, self.text)
    #         # [-1, 1] scaled to [0, 1]
    #         score = logits_per_image[0].cpu().numpy()/100
    #         scaledScore = (score + 1) / 2
    #     return scaledScore

    def getScore(self, img):
        img_pil = Image.fromarray(img)
        image = self.vis_processors["eval"](img_pil).unsqueeze(0).to(self.device)
        sample = {"image": image, "text_input": self.text}
        features_image = self.model.extract_features(sample, mode="image")
        features_text = self.model.extract_features(sample, mode="text")
        similarity = features_image.image_embeds_proj[:,0,:] @ features_text.text_embeds_proj[:,0,:].t()
        # [-1, 1] scaled to [0, 1]
        score = (similarity[0].cpu().numpy() + 1) / 2
        return score

if __name__ == "__main__":

    action = "doing the squat"
    prompts = [
        f"Break down the action into just two subsequential poses that must be achieved to complete the action.\n\
        Assume the intitial pose is standing. Just list the them without any comments.\n\
        action: jumping jacks \n\
        subposes: Arms and legs spread wide, forming an X shape / Arms and legs brought back together at the midline, with feet together and arms at the sides \n\
        action: doing the squat \n\
        subposes:"  
    ]
    # select LLM
    llm = GPT(model="gpt-3.5-turbo-instruct")
    actionList = llm.complete(prompts).split("/")

    # select VLM and model version
    # print(model_zoo) to check for available models
    vlm = VLM(model_name = "blip_feature_extractor",
              model_version = 'base', 
              actionText=actionList
            )

    cap = cv2.VideoCapture('./videos/squat.mp4')
    totalScore = []
    stTime = time.time()
    while cap.isOpened():
        success, img = cap.read()
        if not success:
            # print("Ignoring empty camera frame.")
            # If loading a video, use 'break' instead of 'continue'.
            break
        r1, r2 = vlm.getScore(img)[0], vlm.getScore(img)[1]
        totalScore.append(vlm.getScore(img))


    print("===================================")
    print("first sub-pose", actionList[0])
    print("second sub-pose", actionList[1])

    plt.plot(np.arange(len(totalScore)), totalScore)
    plt.legend(["squating", "standing"])
    plt.xlabel("Frame")
    plt.ylabel("Similarity Score")
    plt.title("Similarity Score of each frame")
    plt.savefig("similarity.png")
    plt.show()
    
    # avgScore = totalScore.mean()
    # print("===================================")
    # print(totalScore)
    # print(totalScore[0])
    # print("Average simliarity score: {:.4f}".format(avgScore))
    # endTime = time.time()
    # print("FPS: {:.2f}".format(i / (endTime - stTime)))
    # print("===================================")