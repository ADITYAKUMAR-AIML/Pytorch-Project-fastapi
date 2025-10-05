from fastapi import FastAPI, UploadFile, File, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from PIL import Image
import torch, io, base64
import torch.nn.functional as F
from torchvision import transforms, models
from torchvision.models import MobileNet_V3_Large_Weights

app = FastAPI() #app is your FastAPI application.
templates = Jinja2Templates(directory="templates") #templates tells FastAPI that your HTML files are stored in the folder named “templates”.

# model setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
weights = MobileNet_V3_Large_Weights.DEFAULT #load up the default weights for our model.
model = models.mobilenet_v3_large(weights=weights) #now create the model with those (trained) weights.
model.eval().to(device)

"""""device → decides whether to use GPU (cuda) or CPU.

weights → load pretrained weights for MobileNetV3.

model → create the MobileNetV3 model.

model.eval() → put it in evaluation mode (no training).

to(device) → move the model to GPU if available.

This means the model is ready to classify images."""

# ✅ Get ImageNet class labels
imagenet_labels = weights.meta["categories"] #To get all of the labels.

# The transformation is based on the these different transforms.
preprocess = transforms.Compose([ 
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std =[0.229, 0.224, 0.225]),
])


@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/predict", response_class=HTMLResponse)
async def predict(request: Request, file: UploadFile = File(...)):
    img_bytes = await file.read()
    img = Image.open(io.BytesIO(img_bytes)).convert("RGB")

    x = preprocess(img).unsqueeze(0).to(device) #transform the image to the tensor with other transformations
    with torch.no_grad():
        logits = model(x) #get the logits, by passing the inputs to our model.
        probs = F.softmax(logits, dim=1) #get the probabilities rather than logits
        topk = torch.topk(probs, k=5) #get the top 5 most likely classes

    top_probs = topk.values[0].cpu().tolist()
    top_idxs  = topk.indices[0].cpu().tolist()

    # ✅ Map IDs to human-readable labels
    top_labels = [imagenet_labels[idx] for idx in top_idxs]
    
    #To convert the img_byts into the Human redable image.
    b64 = base64.b64encode(img_bytes).decode("utf-8")
    mime = file.content_type or "image/jpeg"
    data_url = f"data:{mime};base64,{b64}"

    # ✅ Pass names + probabilities
    rows = zip(top_labels, top_probs)

    return templates.TemplateResponse("result.html", {
        "request": request,
        "image_data": data_url,
        "rows": rows
    })
