from omnixai.explainers.vision import VisionExplainer
from omnixai.visualization.dashboard import Dashboard
import torch
from torchvision import transforms
from PIL import Image as PilImage
from omnixai.data.image import Image

img_size = 224
model = torch.load(r'models\resnet18_is224_bs4_e10_i10000_resize0.5_contrastReduce1-20_num_image10000.pth').to('cpu')
# Enable gpu mode, if cuda available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

test_img =  Image(PilImage.open(r"images_resize0.5_contrastReduce1-20_num_image10000\part_whole_flip_test\1\s_min+3.png").convert('RGB'))

transform = transforms.Compose([
        transforms.Resize(size=img_size),
        transforms.ToTensor(),
        ])
preprocess = lambda ims: torch.stack([transform(im.to_pil()) for im in ims])

explainer = VisionExplainer(
    explainers=["gradcam", "lime", "ig", "ce", "feature_visualization"],
    mode="classification",
    model=model,                   # An image classification model, e.g., ResNet50
    preprocess=preprocess,         # The preprocessing function
    # postprocess=postprocess,       # The postprocessing function
    params={
        # Set the target layer for GradCAM
        "gradcam": {"target_layer": model.layer3[-1]},
        # Set the objective for feature visualization
        "feature_visualization": 
          {"objectives": [{"layer": model.layer3[-1], "type": "channel", "index": list(range(6))}]}
    },
)
# Generate explanations of GradCAM, LIME, IG and CE
local_explanations = explainer.explain(test_img)
# Generate explanations of feature visualization
global_explanations = explainer.explain_global()

model.eval()
# Load the class names
idx2label = ['0_plus', '1_min']
input_img = transform(test_img.to_pil()).unsqueeze(dim=0)
probs_top_2 = torch.nn.functional.softmax(model(input_img), dim=1).topk(2)
r = tuple((p, c, idx2label[c]) for p, c in
          zip(probs_top_2[0][0].detach().numpy(), probs_top_2[1][0].detach().numpy()))
print(r)

# Launch the dashboard
dashboard = Dashboard(
    instances=test_img,
    local_explanations=local_explanations,
    global_explanations=global_explanations
)
dashboard.show()