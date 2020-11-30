import sys
import os
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image


def main():
    m = sys.argv[1]
    image_dir = sys.argv[2]
    categories = sys.argv[3]
    places = ["Large Hall", "Studio", "Medium Hall", "Outdoor", "Small Space", "Home Entryway", "Living Room"]
    print(places)
    
    model = models.resnet50(num_classes=365)
    c = torch.load(m, map_location="cpu")
    state_dict = {k.replace("module.", ""): v for k, v in c["state_dict"].items()}
    model.load_state_dict(state_dict)
    model.eval()

    with open(categories) as infile:
        c = infile.read().split("\n")
        c = ["_".join(l.split()[0].split("/")[2:]) for l in c]

    t = transforms.Compose([transforms.Scale([224, 224], Image.BICUBIC), transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    examples = []
    images = list(filter(lambda f : os.path.splitext(f)[1] in (".jpg", ".png"), os.listdir(image_dir)))
    n_images = len(images)
    for i, f in enumerate(images):
        f = os.path.join(image_dir, f)
        img = Image.open(f).convert("RGB")
        z = t(img).unsqueeze(0)
        y = c[int(torch.nn.functional.softmax(model(z)).argmax())]
        print("%d/%d" % (i, n_images), y)
        examples.append(y)

    examples_d = set(examples)
    example_counts = [(name, examples.count(name)) for name in examples_d]
    example_counts.sort(key=lambda example : -example[1])
    for example in example_counts:
        print(example[0], example[1])



if __name__ == "__main__":
    main()