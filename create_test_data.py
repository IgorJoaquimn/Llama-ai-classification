import pandas as pd
import os

data = {
    "transcript": [
        "In this video, we explore the new features of GPT-4o, including its multimodal capabilities and how it can be used for real-time translation and coding assistance. Sam Altman discusses the future of OpenAI and the impact of large language models on the job market.",
        "Today we are reviewing the latest NVIDIA RTX 6000 Ada Generation GPU. We'll look at its performance in rendering, AI training, and how it compares to the previous generation A6000. This card is a beast for workstation tasks.",
        "This is a tutorial on how to bake a chocolate cake from scratch. We will go through the ingredients like flour, cocoa powder, and eggs, and then show the step-by-step process of mixing and baking."
    ]
}

df = pd.DataFrame(data)
os.makedirs("data", exist_ok=True)
df.to_parquet("data/test_input.parquet")
print("Created data/test_input.parquet with 3 samples.")
