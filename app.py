import gradio as gr
import tensorflow as tf
import numpy as np
from PIL import Image
from tensorflow.keras.applications.densenet import preprocess_input

MODEL_PATH = "best_densenet121_pneumonia.keras"
IMG_SIZE = 224

model = tf.keras.models.load_model(MODEL_PATH)

def predict(img):
    if img is None:
        return "Please upload an image."

    img = img.convert("RGB")
    img = img.resize((IMG_SIZE, IMG_SIZE))

    x = np.array(img).astype("float32")
    x = x.reshape(1, IMG_SIZE, IMG_SIZE, 3)

    prob = float(model.predict(x, verbose=0)[0][0])

    # 0 → PNEUMONIA, 1 → NORMAL
    if prob >= 0.5:
        return f"NORMAL\n"
    else:
        return f"PNEUMONIA\n"

with gr.Blocks(
    theme=gr.themes.Soft(),
    title="Pneumonia Classifier"
) as demo:

    gr.Markdown(
        """
        # 🩺 Pneumonia Classifier
        Upload a **chest X-ray** image to classify:
        - **NORMAL**
        - **PNEUMONIA**
        """
    )

    with gr.Row():
        with gr.Column(scale=1):
            image_input = gr.Image(
                type="pil",
                label="Chest X-ray Image",
                height=350
            )
            submit_btn = gr.Button("Predict", variant="primary")

        with gr.Column(scale=1):
            output = gr.Textbox(
                label="Result",
                lines=3
            )

    submit_btn.click(
        fn=predict,
        inputs=image_input,
        outputs=output
    )

demo.launch()
