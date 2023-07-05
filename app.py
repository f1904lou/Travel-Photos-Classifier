import gradio as gr
from fastai.vision.all import *
learn = load_learner('Photo_classification_model.pkl')

labels = learn.dls.vocab
labels = [labels.capitalize() for labels in labels]
def predict(img):
    img = PILImage.create(img)
    pred,pred_idx,probs = learn.predict(img)
    return {labels[i]: float(probs[i]) for i in range(len(labels))}

title = "Travel Photos Classifier"
description = "A travel photo classifier trained with fastai. Created as a demo for Gradio and HuggingFace Spaces."
interpretation='default'
enable_queue=True

gr.Interface(fn=predict,inputs=gr.inputs.Image(shape=(512, 512)),outputs=gr.outputs.Label(num_top_classes=3),title=title,description=description,interpretation=interpretation,enable_queue=enable_queue).launch()
