from gluoncv.model_zoo import get_model

def get_pretrained_model(model_name):
    if model_name == 'slowfast':
        model = get_model('slowfast_8x8_resnet101_kinetics400', pretrained=True)
    elif model_name == 'i3d_ucf':
        model = get_model('i3d_resnet50_v1_ucf101', pretrained=True)
    elif model_name == 'i3d_hmdb51':
        model = get_model('i3d_resnet50_v1_hmdb51', pretrained=True)
    else:
        raise Exception('No such model!')
    return model

def correctly_classified(dataset, model, x, y):
    transform_fn = get_transform(dataset)
    normalized_x = transform_fn(x)
    preds = model(normalized_x)
    post_act = torch.nn.Softmax(dim=1)
    preds = post_act(preds)
    pred_classes = preds.topk(k=5).indices[0]
    if dataset == 'kinetics400':
        pred_class_names = [kinetics_id_to_classname[int(i)] for i in pred_classes]
        id_to_classname = kinetics_id_to_classname
    elif dataset == 'ucf101':
        pred_class_names = [ucf101_id_to_classname[int(i)] for i in pred_classes]
        id_to_classname = ucf101_id_to_classname
    print("########################################")
    print("Top names:  %s" % "\n\t\t\t".join(pred_class_names))
    print(f'Top labels: {np.array2string(np.array(pred_classes.cpu()))}')
    print(f'Original name: {id_to_classname[int(y)]}')
    print(f'Original label: {int(y)}')
    print("########################################")
    return y in pred_classes