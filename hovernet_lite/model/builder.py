import torch
from hovernet_lite.model.hovernet import HoVerNet


def create_model(mode='fast', input_ch=3, num_of_nuc_types=None, freeze=False):
    assert mode in ['original', 'fast'], f"Unknown Model Mode {mode}"
    return HoVerNet(mode=mode, input_ch=input_ch, nr_types=num_of_nuc_types, freeze=freeze)


def convert_pytorch_checkpoint(net_state_dict):
    variable_name_list = list(net_state_dict.keys())
    is_in_parallel_mode = all(v.split(".")[0] == "module" for v in variable_name_list)
    if is_in_parallel_mode:
        print(
            (
                "WARNING: Detect checkpoint saved in data-parallel mode."
                " Converting saved model to single GPU mode."
            ).rjust(80)
        )
        net_state_dict = {
            ".".join(k.split(".")[1:]): v for k, v in net_state_dict.items()
        }
    return net_state_dict


def load_model(path, mode, num_of_nuc_types):
    print("Torch model loading started.")
    net = create_model(input_ch=3, num_of_nuc_types=num_of_nuc_types, freeze=False, mode=mode)
    print("Model has been created")
    saved_state_dict = torch.load(path)["desc"]
    print("Conversion from tf to pytorch started")
    saved_state_dict = convert_pytorch_checkpoint(saved_state_dict)
    print("Conversion from tf to pytorch ended")

    net.load_state_dict(saved_state_dict, strict=True)
    print("Model weights have been loaded")
    net = torch.nn.DataParallel(net)
    net = net.to("cuda")
    print("Model sent to cuda")
    return net
