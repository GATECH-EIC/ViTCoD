import torch
from timm.models import create_model
from fvcore.nn import FlopCountAnalysis, flop_count_table, flop_count_str


if __name__ == '__main__':
    model = create_model(
    'deit_base_patch16_224',
    pretrained=False)
    model.eval()

    input = torch.rand(1, 3, 224, 224)

    flop = FlopCountAnalysis(model, input)
    print(flop_count_table(flop, max_depth=4))
    print(flop_count_str(flop))
    print(flop.total())