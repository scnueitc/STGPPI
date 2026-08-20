from torch.optim import AdamW


def get_optimizer(model, lr=1e-3, weight_decay=1e-2):
    
    wd_params = []
    no_wd_params = []

    for param in model.parameters():
        if param.requires_grad:
           
            if param.ndim < 2:
                no_wd_params.append(param)
            else:
                wd_params.append(param)

    optimizer_grouped_parameters = [
        {'params': wd_params, 'weight_decay': weight_decay},
        {'params': no_wd_params, 'weight_decay': 0.0},
    ]

    optimizer = AdamW(optimizer_grouped_parameters, lr=lr)
    return optimizer

