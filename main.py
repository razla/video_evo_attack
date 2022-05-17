import torch

from utils import get_model, get_dataset, parse_main, correctly_classified, print_summary
from island_attack import EvoAttack
# from regular_attack import EvoAttack

device = 'cuda' if torch.cuda.is_available() else 'cpu'

if __name__ == '__main__':
    model_name, dataset, eps, top_k, n_pop, n_gen, n_videos, n_tournament, n_iter, target = parse_main()
    dataloader, videos = get_dataset(dataset, n_videos * 2)
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

        if correctly_classified(dataset, model, x, y) and count < n_videos:
            count += 1
            adv, n_queries = EvoAttack(dataset=dataset,
                                       model=model,
                                       x=x,
                                       y=y,
                                       n_gen=n_gen,
                                       n_pop=n_pop,
                                       n_tournament=n_tournament,
                                       eps=eps,
                                       top_k=top_k,
                                       target=target).generate()
            if not isinstance(adv, type(None)):
                success_count += 1
            evo_queries.append(n_queries)

    print_summary(dataset, model_name, n_videos, n_pop, n_gen, n_tournament, eps, success_count / count, evo_queries)