import torch.nn.functional as F
from operator import itemgetter
import numpy as np
import random
import torch
from utils import normalize, print_success, print_failure, save_video

device = 'cuda' if torch.cuda.is_available() else 'cpu'

class EvoAttack():
    def __init__(self, dataset, model, x, y, n_gen=1000, n_pop=40, n_tournament=35, eps=0.3, top_k = 5, alpha = 0.05, target=None):
        self.dataset = dataset
        self.model = model
        self.x = x
        self.y = y
        self.p_init = 0.1
        self.n_gen = n_gen
        self.n_pop = n_pop
        self.n_tournament = n_tournament
        self.eps = eps
        self.top_k = top_k
        self.alpha = alpha
        self.best_x_hat = None
        self.bad_x_hat = None
        self.queries = 0
        self.target = target
        self.rho = 1
        self.min_ball = torch.tile(torch.maximum(self.x - eps, torch.tensor(0)), (1, 1))
        self.max_ball = torch.tile(torch.minimum(self.x + eps, torch.tensor(1)), (1, 1))

    def generate(self):
        save_video(self.x, self.y, f'orig_{self.y.item()}.avi')
        gen = 0
        cur_pop = self.init()
        while not self.termination_condition(cur_pop, gen):
            self.fitness(cur_pop)

            print(f'Gen #{gen} best fitness: {min(cur_pop, key=itemgetter(1))[1]:.5f}')

            new_pop = []
            elite = self.elitism(cur_pop)

            new_pop.append([elite, np.inf])
            for i in range((self.n_pop - 1) // 3):
                parent1 = self.selection(cur_pop)
                parent2 = self.selection(cur_pop)
                offspring1, offspring2 = self.crossover(parent1, parent2)
                mut_offspring1 = self.per_frame_mutation((offspring1, np.inf))
                mut_offspring2 = self.per_frame_mutation((offspring2, np.inf))
                offspring1 = self.project(offspring1)
                new_pop.append([offspring1, np.inf])
                new_pop.append([mut_offspring1, np.inf])
                new_pop.append([mut_offspring2, np.inf])

            cur_pop = new_pop
            gen += 1
            self.rho *= 0.9

        return self.best_x_hat, self.queries

    def crossover(self, parent1, parent2):
        frames = self.x.shape[2]
        frame_shape = (self.x.shape[1], self.x.shape[3], self.x.shape[4])
        offspring1 = parent1[0].clone()
        offspring2 = parent2[0].clone()
        for f in range(frames):
            flat_parent1 = parent1[0][0,:,f].flatten()
            flat_parent2 = parent2[0][0,:,f].flatten()
            i = np.random.randint(0, len(flat_parent1))
            j = np.random.randint(i, len(flat_parent1))
            cat_frame1 = torch.cat([flat_parent1[:i], flat_parent2[i:j], flat_parent1[j:]], dim = 0)
            cat_frame2 = torch.cat([flat_parent2[:i], flat_parent1[i:j], flat_parent2[j:]], dim = 0)
            offspring1[0, :, f] = cat_frame1.reshape(frame_shape)
            offspring2[0, :, f] = cat_frame2.reshape(frame_shape)
        offspring1 = self.project(offspring1)
        offspring2 = self.project(offspring2)
        return offspring1, offspring2

    def elitism(self, cur_pop):
        elite = min(cur_pop, key=itemgetter(1))[0]
        return elite

    def selection(self, cur_pop):
        selection = [random.choice(cur_pop) for i in range(self.n_tournament)]
        best = min(selection, key=itemgetter(1))
        return best

    def per_frame_mutation(self, x_hat):
        p = self.p_selection(self.p_init, self.queries, self.n_gen * self.n_pop)
        channels = x_hat[0].shape[1]
        frames = x_hat[0].shape[2]
        height = x_hat[0].shape[3]
        width = x_hat[0].shape[4]
        n_features = channels * height * width
        s = int(round(np.sqrt(p * n_features / channels)))
        s = min(max(s, 1), height - 1)  # at least c x 1 x 1 window is taken and at most c x h-1 x h-1
        for f in range(frames):
            center_h = np.random.randint(0, height - s)
            center_w = np.random.randint(0, width - s)
            x_curr_window = x_hat[0][0, :, f, center_h:center_h + s, center_w:center_w + s]
            for i in range(channels):
                x_curr_window[i] += np.random.choice([-2 * self.eps, 2 * self.eps]) * torch.ones(x_curr_window[i].shape).to(device)
            x_hat[0][0, :, f, center_h:center_h + s, center_w: center_w + s] = x_curr_window
        x_hat = self.project(x_hat[0])
        return x_hat

    def predict(self, x):
        with torch.no_grad():
            probs = F.softmax(self.model(x), dim=1)[0]
        return probs

    def regularization(self, x_hat):
        delta = torch.abs(x_hat - self.x)
        np_delta = np.array(delta.cpu().flatten())
        reg = sum(max(0, x - self.rho) for i, x in enumerate(np_delta))
        return reg

    def fitness(self, cur_pop):
        for i in range(len(cur_pop)):
            x_hat, fitness = cur_pop[i]
            def_x_hat = x_hat.clone()
            reg = self.regularization(def_x_hat)
            n_x_hat = normalize(def_x_hat)
            probs = self.predict(n_x_hat)
            preds_not_y = torch.tensor([x for i, x, in enumerate(probs) if i != self.y])
            objective = max(0, (probs[self.y] - max(preds_not_y).item()).item())
            cur_pop[i] = [x_hat, objective + reg]

    def get_labels(self, x_hat):
        p_x_hat = self.project(x_hat.clone())
        n_x_hat = normalize(p_x_hat)
        probs = self.predict(n_x_hat)
        pred_classes = probs.topk(k=self.top_k).indices[0]
        return pred_classes

    def termination_condition(self, cur_pop, gen):
        if gen == self.n_gen:
            print_failure(self.model, self.queries, self.bad_x_hat, self.y)
            return True
        for [x_hat, _] in cur_pop:
            y_preds = self.get_labels(x_hat)
            self.queries += 1
            if self.y not in y_preds:
                self.best_x_hat = self.project(x_hat)
                print_success(self.model, self.queries, self.best_x_hat, self.y)
                save_video(self.best_x_hat, self.y, f'good_{self.y.item()}.avi')
                return True
            elif isinstance(self.bad_x_hat, type(None)):
                self.bad_x_hat = self.project(x_hat)
                save_video(self.bad_x_hat, self.y, f'bad_{self.y.item()}.avi')
        return False

    def project(self, x_hat):
        projected_x_hat = torch.clip(x_hat, self.min_ball, self.max_ball)
        return projected_x_hat

    def init(self):
        cur_pop = []
        for i in range(self.n_pop):
            x_hat = self.x.clone()
            x_hat = self.vertical_mutation(x_hat)
            cur_pop.append([x_hat, np.inf])
        return cur_pop

    def vertical_mutation(self, x_hat):
        channels = self.x.shape[1]
        frames = self.x.shape[2]
        height = self.x.shape[3]
        width = self.x.shape[4]
        for c in range(channels):
            for f in range(frames):
                for w in range(width):
                    x_hat[0, c, f, :, w] += torch.tensor(self.eps * np.random.choice([-1, 1]) * np.ones((height))).to(device)
        x_hat = self.project(x_hat)
        return x_hat

    def p_selection(self, p_init, it, n_iters):
        """ Piece-wise constant schedule for p (the fraction of pixels changed on every iteration). """
        it = int(it / n_iters * 10000)

        if 10 < it <= 50:
            p = p_init / 2
        elif 50 < it <= 200:
            p = p_init / 4
        elif 200 < it <= 500:
            p = p_init / 8
        elif 500 < it <= 1000:
            p = p_init / 16
        elif 1000 < it <= 2000:
            p = p_init / 32
        elif 2000 < it <= 4000:
            p = p_init / 64
        elif 4000 < it <= 6000:
            p = p_init / 128
        elif 6000 < it <= 8000:
            p = p_init / 256
        elif 8000 < it <= 10000:
            p = p_init / 512
        else:
            p = p_init

        return p