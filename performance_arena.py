#!/usr/bin/env python3

import math
from collections import defaultdict
from functools import partial
from timeit import repeat

import evalica
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from tqdm.auto import tqdm
from rating_systems import compute_bt, compute_bootstrap_bt


def chatbot_arena_elo(battles, K=4, SCALE=400, BASE=10, INIT_RATING=1000):
    rating = defaultdict(lambda: INIT_RATING)

    for rd, model_a, model_b, winner in battles[["model_a", "model_b", "winner"]].itertuples():
        ra = rating[model_a]
        rb = rating[model_b]
        ea = 1 / (1 + BASE ** ((rb - ra) / SCALE))
        eb = 1 / (1 + BASE ** ((ra - rb) / SCALE))
        if winner == "model_a":
            sa = 1
        elif winner == "model_b":
            sa = 0
        elif winner == "tie" or winner == "tie (bothbad)":
            sa = 0.5
        else:
            raise Exception(f"unexpected vote {winner}")
        rating[model_a] += K * (sa - ea)
        rating[model_b] += K * (1 - sa - eb)

    return rating


def arena_hard_bradley_terry(df, SCALE=400, BASE=10, INIT_RATING=1000):
    models = pd.concat([df["model_a"], df["model_b"]]).unique()
    models = pd.Series(np.arange(len(models)), index=models)

    df = pd.concat([df, df], ignore_index=True)
    p = len(models.index)
    n = df.shape[0]

    X = np.zeros([n, p])
    X[np.arange(n), models[df["model_a"]]] = +math.log(10)
    X[np.arange(n), models[df["model_b"]]] = -math.log(10)

    Y = np.zeros(n)
    Y[df["winner"] == "model_a"] = 1.0

    tie_idx = (df["winner"] == "tie") | (df["winner"] == "tie (bothbad)")
    tie_idx[len(tie_idx) // 2:] = False
    Y[tie_idx] = 1.0

    lr = LogisticRegression(fit_intercept=False, penalty=None, tol=1e-8)
    lr.fit(X, Y)

    elo_scores = SCALE * lr.coef_[0] + INIT_RATING

    return pd.Series(elo_scores, index=models.index).sort_values(ascending=False)

def fast_arena_bradley_terry(df, SCALE=400, BASE=10, INIT_RATING=1000):
    return compute_bt(df, base=BASE, scale=SCALE, init_rating=INIT_RATING)

def fast_arena_bootstrap_bradley_terry(df, num_round, SCALE=400, BASE=10, INIT_RATING=1000):
    return compute_bootstrap_bt(df, num_round=num_round, base=BASE, scale=SCALE, init_rating=INIT_RATING)

def main():
    df_arena = pd.read_json("clean_battle_20240814_public.json")
    df_arena = df_arena[df_arena["anony"]]
    df_arena = df_arena[df_arena["dedup_tag"].apply(lambda x: x.get("sampled", False))]
    df_arena["evalica"] = df_arena["winner"].map({
        "model_a": evalica.Winner.X,
        "model_b": evalica.Winner.Y,
        "tie": evalica.Winner.Draw,
        "tie (bothbad)": evalica.Winner.Draw,
    })
    df_arena = df_arena[~df_arena["evalica"].isna()]

    results = []

    REPETITIONS = 10
    NUMBER = 10

    with tqdm(total=5) as pbar:
        arena_elo_time = repeat(
            partial(chatbot_arena_elo, df_arena),
            repeat=REPETITIONS, number=NUMBER,
        )
        results.append(("elo", "arena", arena_elo_time))
        pbar.update()

        # hard_arena_bt_time = repeat(
        #     partial(arena_hard_bradley_terry, df_arena),
        #     repeat=REPETITIONS, number=1,
        # )
        # results.append(("bradley_terry", "arena", hard_arena_bt_time))
        # pbar.update()

        fast_arena_bt_time = repeat(
            partial(fast_arena_bradley_terry, df_arena),
            repeat=REPETITIONS, number=NUMBER,
        )
        results.append(("bradley_terry", "fast arena", fast_arena_bt_time))
        pbar.update()

        fast_arena_bootstrap_bt_time = repeat(
            partial(fast_arena_bootstrap_bradley_terry, df_arena, NUMBER),
            repeat=REPETITIONS, number=1,
        )
        results.append(("bootstrap bradley_terry", "fast arena", fast_arena_bootstrap_bt_time))
        pbar.update()

        evalica_elo_time = repeat(
            partial(evalica.elo, df_arena["model_a"], df_arena["model_b"], df_arena["evalica"]),
            repeat=REPETITIONS, number=NUMBER,
        )
        results.append(("elo", "evalica", evalica_elo_time))
        pbar.update()

        evalica_bt_time = repeat(
            partial(evalica.bradley_terry, df_arena["model_a"], df_arena["model_b"], df_arena["evalica"]),
            repeat=REPETITIONS, number=NUMBER,
        )
        results.append(("bradley_terry", "evalica", evalica_bt_time))
        pbar.update()

    df_results = pd.DataFrame(results, columns=["algorithm", "solver", "time"])
    df_results = df_results.explode("time")
    df_results = df_results.reset_index(drop=True)
    df_results.to_csv("performance_arena.csv", index=False)

    grouped = df_results.groupby(['algorithm', 'solver'])['time'].agg(['mean', 'std']).reset_index()
    grouped['time_stats'] = grouped['mean'].round(2).astype(str) + " ± " + grouped['std'].round(2).astype(str)
    print(grouped[['algorithm', 'solver', 'time_stats']])


if __name__ == "__main__":
    main()