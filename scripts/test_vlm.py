
import ollama
import time

model = "qwen2.5vl:3b"
image = "data/keyframes/scene_003_keyframe.jpg"

start = time.perf_counter()

response = ollama.chat(
    model=model,
    messages=[
        {
            "role": "user",
            "content": (
                "Describe this image. Identify the visible "
                "objects, people, activities and background. "
                "Do not guess details that cannot be seen."
            ),
            "images": [image]
        }
    ],
    options={
        "temperature": 0,
        "num_predict": 100
    }
)

elapsed = time.perf_counter() - start

print("\nDescription:")
print(response["message"]["content"])

print(f"\nInference time: {elapsed:.2f} seconds")
