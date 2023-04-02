import random
from itertools import combinations
from multiprocessing import Pool

circles = [1, 2, 3, 4, 5, 7, 8, 10, 11, 12, 13, 14]
triangles = [1, 2, 3, 4, 5, 7, 8, 10, 11, 12, 13, 14]
crosses = [1, 2, 3, 5, 7, 10, 11, 13, 14]
squares = [1, 2, 3, 5, 7, 10, 11, 13, 14]
stars = [1, 2, 3, 4, 5, 7, 8]
whots = [20, 20, 20, 20]

stores = {
    'circle': circles,
    'triangle': triangles,
    'cross': crosses,
    'sqaure': squares,
    'star': stars,
    'whot': whots
}


def deck():
    store = []
    for i in list(stores.keys()):
        for j in stores[i]:
            store.append((i, j))
    return store


def stacks(n, k):
    stack = {}
    fake = deck().copy()
    for i in range(n + 1):
        new = random.choices(fake, k=k)
        new.sort()
        stack[f'player{i}'] = new
        for j in new:
            try:
                fake.remove(j)
            except:
                pass
    stack['start'] = fake[0]

    return stack


def nextplay(cards, card):
    valid = []
    for i in cards:
        if card[1] == 14:
            return 'go market'
        if card[1] == 8:
            return 'hold on'

        if i[0] == card[0] or i[1] == card[1]:
            return i
    return 'go market'


def process_data(box):
    data = []
    for ite in box:
        data.append(
            [*[" ".join([str(card) for card in a]) for a in box[0]['cards']]] +
            [" ".join([str(card) for card in ite["played"]]), " ".join([str(card) for card in ite["action"]])]
        )
    return data


if __name__ == '__main__':
    a = combinations(deck(), 4)
    combos = []
    for _ in a:
        c = [d for d in _]
        c.sort()
        combos.append(_)

    box = []
    for jac in combos:
        for play in deck():
            box.append({
                'cards': jac,
                'action': nextplay(jac, play),
                'played': play
            })

    with Pool() as p:
        results = p.map(process_data, [box[i:i+1000] for i in range(0, len(box), 1000)])

    data = [d for r in results for d in r]

    import pandas as pd
    df = pd.DataFrame(data, columns=[*[f'Card {i + 1}' for i in range(4)] + ['Played', 'Action']])
    df.to_csv("data.csv")
