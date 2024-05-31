import numpy as np
import onnx
import torch

from swinface import SwinFace, SwinFaceCfg, get_swinface_parser

def convert_onnx(net, path_module, output, opset=11, simplify=False):
    assert isinstance(net, torch.nn.Module)
    img = np.random.randint(0, 255, size=(112, 112, 3), dtype=np.int32)
    img = img.astype(np.float)
    img = (img / 255. - 0.5) / 0.5  # torch style norm
    img = img.transpose((2, 0, 1))
    img = torch.from_numpy(img).unsqueeze(0).float()


    torch.onnx.export(net, img, output, input_names=["data"], keep_initializers_as_inputs=False, verbose=False, opset_version=opset)
    model = onnx.load(output)
    graph = model.graph
    graph.input[0].type.tensor_type.shape.dim[0].dim_param = 'None'
    if simplify:
        from onnxsim import simplify
        model, check = simplify(model)
        assert check, "Simplified ONNX model could not be validated"
    onnx.save(model, output)
    print(f"ONNX Model saved to {output}")

    
if __name__ == '__main__':
    import os
    import argparse
    swinface_model_parser = get_swinface_parser()
    swinface_args = swinface_model_parser.parse_args()
    swinface = SwinFace(SwinFaceCfg(swinface_args))

    parser = argparse.ArgumentParser(description='PyTorch to onnx')
    parser.add_argument('--input', type=str, default='src/swinface_project/checkpoint_step_79999_gpu_0.pt', help='input backbone.pth file or path')
    parser.add_argument('--output', type=str, default='swinface.onnx', help='output onnx path')
    parser.add_argument('--simplify', type=bool, default=False, help='onnx simplify')
    args = parser.parse_args()
    input_file = args.input
    if os.path.isdir(input_file):
        input_file = os.path.join(input_file, "model.pt")
    assert os.path.exists(input_file)
    print(args)

    if args.output is None:
        args.output = os.path.join(os.path.dirname(args.input), "model.onnx")
    convert_onnx(swinface.model, input_file, args.output, simplify=args.simplify)
