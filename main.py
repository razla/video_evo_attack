import numpy as np
import torch

from utils import get_model, get_dataset, parse, correctly_classified, print_summary
from attack import EvoAttack

device = 'cuda' if torch.cuda.is_available() else 'cpu'

if __name__ == '__main__':

    model_name, dataset, eps, n_pop, n_gen, n_videos, n_tournament, n_frames, n_iter = parse()

    dataloader, videos = get_dataset(n_videos * 2)
    model = get_model(model_name)
    model = model.to(device)

    count = 0
    success_count = 0
    evo_queries = []
    for x, y in dataloader:
        if count == n_videos:
            break

        x = x.to(device)
        y = y.to(device)

        if correctly_classified(model, x, y) and count < n_videos:
            count += 1
            adv, n_queries = EvoAttack(dataset=dataset, model=model, x=x, y=y, n_gen=n_gen, n_pop=n_pop,n_tournament=n_tournament, eps=eps).generate()
            if not isinstance(adv, type(None)):
                success_count += 1
            evo_queries.append(n_queries)

    print_summary(dataset, model_name, n_videos, n_pop, n_gen, n_tournament, n_frames, eps, success_count / count, evo_queries)