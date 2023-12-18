import torch
import cv2
from PIL import Image
import time 
# from gpt import GPT
import numpy as np

import matplotlib.pyplot as plt
from lavis.models import model_zoo, load_model_and_preprocess

class VLM():
    def __init__(self, model_name = 'clip_feature_extractor',  model_version = "ViT-L-14", actionText = ["walking"], use_itm = False):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model_name = model_name
        self.model, self.vis_processors, self.txt_processors = load_model_and_preprocess(name = self.model_name, 
                                                                model_type = model_version, 
                                                                is_eval=True,
                                                                device=self.device)
        self._use_itm = use_itm
        # self.text = [self.txt_processors["eval"](f"a human is {c}") for c in actionText]
        self.text = [self.txt_processors["eval"](c) for c in actionText]

    def getScore(self, img):
        img_pil = Image.fromarray(img)
        image = self.vis_processors["eval"](img_pil).unsqueeze(0).to(self.device)
        sample = {"image": image, "text_input": self.text}

        if "clip" in self.model_name:
            clip_features = self.model.extract_features(sample)
            features_image = clip_features.image_embeds_proj
            features_text = clip_features.text_embeds_proj
            similarity = (features_image @ features_text.t())[0].detach().cpu().numpy()
            # # [-1, 1] scaled to [0, 1]
            score = (similarity + 1) / 2
            return score
        
        if self._use_itm:
            itm_scores = np.zeros(len(self.text))
            for i, txt in enumerate(self.text):
                sample = {"image": image, "text_input": txt}
                itm_output = self.model(sample, match_head="itm")
                itm_scores[i] = torch.nn.functional.softmax(itm_output, dim=1)[:, 1].item()
            return itm_scores

        features_image = self.model.extract_features(sample, mode="image")
        features_text = self.model.extract_features(sample, mode="text")
        similarity = (features_image.image_embeds_proj[:,0,:] @ features_text.text_embeds_proj[:,0,:].t())[0].cpu().numpy()
        # [-1, 1] scaled to [0, 1]
        # score = (similarity + 1) / 2
        return similarity

if __name__ == "__main__":
    # # select LLM
    # llm = GPT(model="gpt-3.5-turbo-instruct")
    # action = "doing the squat"
    # prompts = [
    #     f"Break down the action into just two subsequential poses that must be achieved to complete the action.\n\
    #     Assume the intitial pose is standing. Just list the them without any comments.\n\
    #     action: jumping jacks \n\
    #     subposes: Arms and legs spread wide, forming an X shape / Arms and legs brought back together at the midline, with feet together and arms at the sides \n\
    #     action: {action} \n\
    #     subposes:"  
    # ]
    # actionList = llm.complete(prompts).split("/")

    actionList = [
                  "Squatting down with knees bent and hips lowered",
                  "Return to standing position with legs straight and hips raised",
                #   "Hips back and down, knees bent, thighs parallel to the floor",
                #   "Return to standing with hips and knees extended",
                #   "Hips lowered with knees bent and back straight",
                #   "Hips raised with legs straightened and back upright",
                #   "Hips back, knees bent, thighs parallel to the floor",
                #   "Return to standing position with hips and knees fully extended, arms relaxed at sides"
                ]

    # select VLM and model version
    # print(model_zoo) to check for available models
    vlm = VLM(
              model_name = "blip2_image_text_matching",
              model_version = 'pretrain', 
              actionText=actionList,
              use_itm = True,
            )

    cap = cv2.VideoCapture('./videos/test.mp4')
    totalScore = []
    stTime = time.time()
    while cap.isOpened():
        success, img = cap.read()
        if not success:
            # print("Ignoring empty camera frame.")
            # If loading a video, use 'break' instead of 'continue'.
            break
        score = vlm.getScore(img)
        totalScore.append(score)

    print("===================================")
    # print("first sub-pose:", actionList[0], '\n')
    # print("second sub-pose:", actionList[1], '\n')

    plt.plot(np.arange(len(totalScore)), totalScore)
    plt.legend(["squating", "standing"])
    plt.xlabel("Frame")
    plt.ylabel("Similarity Score")
    plt.title("Similarity Score of each frame")
    # plt.savefig("similarity_clip.png")
    plt.show()
    
    # avgScore = totalScore.mean()
    # print("===================================")
    # print(totalScore)
    # print(totalScore[0])
    # print("Average simliarity score: {:.4f}".format(avgScore))
    # endTime = time.time()
    # print("FPS: {:.2f}".format(i / (endTime - stTime)))
    # print("===================================")