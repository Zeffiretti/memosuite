import os
import ffmpeg
from PIL import Image
import torch
import shutil
from transformers import AutoProcessor, AutoModelForVision2Seq

MAX_FRAMES = 8
FRAME_RATE = 1
MODEL_ID = "google/paligemma2-3b-pt-224"

# ReopenDrawer
# VIDEO_PATH = "robosuite/models/assets/demonstrations_private/1755753640_4157922/playback_cut.mp4"
VIDEO_PATH = "robosuite/models/assets/demonstrations_private/1755753640_4157922/playback_bottom_cut.mp4"
# QUESTION = "Which drawer should I open again to find the red cube—the top one or the bottom one?"
QUESTION = "Which drawer should I open again to find the red cube—the white-handle one or the blue-handle one?"

# PushBackCube
# VIDEO_PATH = "robosuite/models/assets/demonstrations_private/1755830752_495001/pushbackcube_cut.mp4"
# QUESTION = (
#     "Wchich direction should I move the cube to place it on the original position at the very begining of the video."
# )


def extract_video_frames(video_path, frame_rate=1):
    if os.path.exists("frames"):
        shutil.rmtree("frames")
    os.makedirs("frames", exist_ok=True)

    print("[INFO] Extracting video frames...")
    ffmpeg.input(video_path).output("frames/frame_%04d.jpg", vf=f"fps={frame_rate}").overwrite_output().run(quiet=True)
    print("[INFO] Frame extraction done.")


def prepare_inputs(processor, question):
    frame_files = sorted(f for f in os.listdir("frames") if f.endswith(".jpg"))
    frame_images = [Image.open(os.path.join("frames", f)) for f in frame_files]

    # 限制帧数
    if len(frame_images) > MAX_FRAMES:
        frame_images = frame_images[:MAX_FRAMES]

    print(f"[INFO] Preparing model input with {len(frame_images)} frames ...")

    # prompt 中插入 <image> 占位符
    prompt = " ".join(["<image>"] * len(frame_images)) + " " + question

    # 注意 images 是 List[Image]，不是嵌套列表
    inputs = processor(
        text=[prompt],
        images=frame_images,
        return_tensors="pt",
        padding=True,
    ).to("cuda")

    return inputs


def run_inference(model, processor, inputs):
    print("[INFO] Running inference...")
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=256)

    answer = processor.decode(outputs[0], skip_special_tokens=True)
    return answer


def main():
    extract_video_frames(VIDEO_PATH, FRAME_RATE)

    print("[INFO] Loading PaliGemma2 model...")
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    model = AutoModelForVision2Seq.from_pretrained(MODEL_ID, torch_dtype=torch.float16, device_map="auto")
    model.eval()

    inputs = prepare_inputs(processor, QUESTION)
    answer = run_inference(model, processor, inputs)

    print("\n===== Model Answer =====")
    print(answer)
    print("========================")


if __name__ == "__main__":
    main()
