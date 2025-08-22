import os
import ffmpeg
from PIL import Image
import torch
import shutil
from transformers import AutoProcessor, AutoModelForVision2Seq


MAX_FRAMES = 8
FRAME_RATE = 1
MODEL_ID = "Qwen/Qwen2.5-VL-3B-Instruct"

# # ReopenDrawer
# VIDEO_PATH = "robosuite/models/assets/demonstrations_private/1755753640_4157922/playback_cut.mp4"
VIDEO_PATH = "robosuite/models/assets/demonstrations_private/1755753640_4157922/playback_bottom_cut.mp4"
QUESTION = "Which drawer should I open again to find the red cube—the top one or the bottom one?"
# QUESTION = "Which drawer should I open again to find the red cube—the white-handle one or the blue-handle one?"

# # PushBackCube
# VIDEO_PATH = "robosuite/models/assets/demonstrations_private/1755830752_495001/pushbackcube_cut.mp4"
# QUESTION = (
#     "Wchich direction should I move the cube to place it on the original position at the very begining of the video."
# )


def extract_video_frames(video_path, frame_rate=1):
    if os.path.exists("frames"):
        shutil.rmtree("frames")
    os.makedirs("frames", exist_ok=False)
    print("[INFO] Extracting video frames...")
    ffmpeg.input(video_path).output("frames/frame_%04d.jpg", vf=f"fps={frame_rate}").overwrite_output().run(quiet=True)
    print("[INFO] Extraction done.")


def prepare_messages(frame_dir, question):
    frames = sorted([f for f in os.listdir(frame_dir) if f.endswith(".jpg")])
    print(f"[INFO] Found {len(frames)} frames.")

    content = []
    for f in frames:
        img_path = os.path.join(frame_dir, f)
        content.append({"type": "image", "image": Image.open(img_path)})
    # 添加问题
    content.append({"type": "text", "text": question})

    messages = [{"role": "user", "content": content}]
    return messages


def main():
    extract_video_frames(VIDEO_PATH, FRAME_RATE)

    print("[INFO] Loading model...")
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    model = AutoModelForVision2Seq.from_pretrained(MODEL_ID, torch_dtype=torch.float16, device_map="auto")
    model.cuda().eval()

    messages = prepare_messages("frames", QUESTION)

    inputs = processor.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt"
    ).to(model.device)

    # 推理
    print("[INFO] Running inference...")
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=100)

    answer = processor.decode(outputs[0][inputs["input_ids"].shape[-1] :])
    print("\n===== Model Answer =====")
    print(answer)
    print("========================")


if __name__ == "__main__":
    main()
