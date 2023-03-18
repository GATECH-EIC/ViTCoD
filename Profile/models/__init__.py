
import sys
sys.path.append("../models")
from linformer import Linformer
from vit import Transformer as ViT_Transformer

def build_model(model_name, model_config, tflite=False):
    if "Linformer" in model_name:
        model = Linformer(dim = model_config["dim"],
                          seq_len = model_config["seq_len"],
                          depth = 12,
                          heads = model_config["heads"],
                          k = min(256, (model_config["seq_len"]//8)//8*8), # k is the key factor of the effiiciency, k <= 256 in https://arxiv.org/pdf/2006.04768.pdf
                          one_kv_head = True,
                          share_kv = True,
                          tflite=tflite)
    elif "ViT_Transformer" in model_name:
        model = ViT_Transformer(dim = model_config["dim"],
                          depth = 12,
                          heads = model_config["heads"],
                          dim_head = model_config["dim"] // model_config["heads"],
                          mlp_dim = model_config["dim"]*4,
                          tflite=tflite)
    else:
        print("To be implemented: {}".format(model_name))
        exit()
    return model