import numpy as np
import torch

from utils import get_model, get_dataset, parse, correctly_classified
from attack import EvoAttack

device = 'cuda' if torch.cuda.is_available() else 'cpu'


if __name__ == '__main__':

    model_name, dataset, eps, n_pop, n_gen, n_videos, n_tournament, n_iter = parse()

    dataloader = get_dataset(n_videos * 2)
    model = get_model(model_name)

    count = 0
    success_count = 0
    evo_queries = []
    videos_indices = []
    for i, (x, y) in enumerate(dataloader):
        if count == n_videos:
            break

        if correctly_classified(model, x, y) and count < n_videos:
            count += 1
            videos_indices.append(i)
            adv, n_queries = EvoAttack(dataset=dataset, model=model, x=x, y=y, n_gen=n_gen, n_pop=n_pop,n_tournament=n_tournament, eps=eps).generate()

            if not isinstance(adv, type(None)):
                success_count += 1
                adv = adv.cpu().numpy()
                if success_count == 1:
                    evo_x_test_adv = adv
                else:
                    evo_x_test_adv = np.concatenate((adv, evo_x_test_adv), axis=0)
            else:
                print('Evolution failed!')
            evo_queries.append(n_queries)

    # x_test, y_test = x_test[images_indices], y_test[images_indices]
    #
    # square_queries, square_adv = [], None
    # for i in range(len(x_test)):
    #
    #     min_ball = torch.tile(torch.maximum(x_test[[i]] - eps, min_pixel_value), (1, 1))
    #     max_ball = torch.tile(torch.minimum(x_test[[i]] + eps, max_pixel_value), (1, 1))
    #
    #     square_adv, square_n_queries = square_attack(dataset, init_model, min_ball, max_ball, x_test[[i]], i, square_adv, n_iter, eps)
    #     square_queries.append(square_n_queries)
    #
    # square_accuracy = compute_accuracy(dataset, init_model, square_adv, y_test, min_pixel_value, max_pixel_value, to_tensor=True, to_normalize=True)
    #
    print('########################################')
    print(f'Summary:')
    print(f'\tDataset: {dataset}')
    print(f'\tModel: {model_name}')
    print(f'\tTournament: {n_tournament}')
    print(f'\tMetric: linf, epsilon: {eps:.4f}')
    # print(f'\tSquare:')
    # print(f'\t\tSquare - test accuracy: {square_accuracy * 100:.4f}%')
    # print(f'\t\tSquare - queries: {square_queries}')
    # print(f'\t\tSquare - queries (median): {int(np.median(square_queries))}')
    print(f'\tEvo:')
    print(f'\t\tEvo - test accuracy: {(1 - (success_count / count)) * 100:.4f}%')
    print(f'\t\tEvo - queries: {evo_queries}')
    print(f'\t\tEvo - queries (median): {int(np.median(evo_queries))}')
    print('########################################')