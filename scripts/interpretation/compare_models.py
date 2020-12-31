import sys
import torch
import torchvision


def main():
    m1, m2 = map(load_model, sys.argv[1:3])
    d = compare_models(m1, m2)
    print("\n%d differences in total." % d)


def load_model(model_path):
    model = torchvision.models.resnet50(num_classes=365)
    c = torch.load(model_path, map_location="cpu")
    state_dict = {k.replace("module.", "").replace("model.", ""): v for k, v in (c["state_dict"].items() if "state_dict" in c else c.items())}
    model.load_state_dict(state_dict)
    model.eval()
    return model


def compare_models(model_1, model_2):
    models_differ = 0
    for key_item_1, key_item_2 in zip(model_1.state_dict().items(), model_2.state_dict().items()):
        if torch.equal(key_item_1[1], key_item_2[1]):
            pass
        else:
            models_differ += 1
            if (key_item_1[0] == key_item_2[0]):
                print("Mismtach found at", key_item_1[0])
            else:
                raise Exception
    if models_differ == 0:
        print("Models match perfectly! :)")
    return models_differ



if __name__ == "__main__":
    main()