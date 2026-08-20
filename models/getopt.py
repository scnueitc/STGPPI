from torch.optim import AdamW


def get_optimizer(model, lr=1e-3, weight_decay=1e-2):
    """
    根据参数维度为模型创建 AdamW 优化器。
    """
    wd_params = []
    no_wd_params = []

    for param in model.parameters():
        if param.requires_grad:
            # 维度小于2的参数（如 bias）不进行权重衰减
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

# 使用示例
# optimizer = get_optimizer(model, lr=1e-3, weight_decay=1e-2)