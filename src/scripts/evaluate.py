from nltk.tokenize import word_tokenize


def calc_distinct_n(n, candidates, print_score: bool = True):
    dict = {}
    total = 0
    candidates = [word_tokenize(candidate) for candidate in candidates]
    for sentence in candidates:
        for i in range(len(sentence) - n + 1):
            ney = tuple(sentence[i : i + n])
            dict[ney] = 1
            total += 1
    score = len(dict) / (total + 1e-16)

    if print_score:
        print(f"***** Distinct-{n}: {score*100} *****")

    return score


def calc_distinct(candidates, print_score: bool = True):
    scores = []
    for i in range(2):
        score = calc_distinct_n(i + 1, candidates, print_score)
        scores.append(score)

    return scores


def read_file(file_name, dec_type="Greedy"):
    f = open(f"results/{file_name}.txt", "r", encoding="utf-8")

    refs = []
    cands = []
    dec_str = f"{dec_type}:"

    for i, line in enumerate(f.readlines()):
        if i == 1:
            _, ppl, _, acc = line.strip("EVAL	Loss	PPL	Accuracy").split()
            print(f"PPL: {ppl}\tAccuracy: {float(acc)*100}%")
        if line.startswith(dec_str):
            exp = line.strip(dec_str).strip("\n")
            cands.append(exp)
        if line.startswith("Ref:"):
            ref = line.strip("Ref:").strip("\n")
            refs.append(ref)

    return refs, cands, float(ppl), float(acc)


if __name__ == "__main__":
    files = [
        "Transformer",
        "Multi-Trs",
        "MoEL",
        "MIME",
        "EmpDG",
        "CEM",
        "CEM_woEMO",
        "CEM_woCOG",
        "CEM_woDiv",
    ]

    best_ppl = 50
    best_acc = 0
    best_dist1 = 0
    best_dist2 = 0
    ppl = ""
    acc = ""
    d1 = ""
    d2 = ""
    for f in files:
        print(f"Evaluating {f}")
        refs, cands, p, a = read_file(f)
        if p < best_ppl:
            ppl = f
            best_ppl = p
        if a > best_acc:
            acc = f
            best_acc = a
        dist_1, dist_2 = calc_distinct(cands)
        if dist_1 > best_dist1:
            d1 = f
            best_dist1 = dist_1
        if dist_2 > best_dist2:
            d2 = f
            best_dist2 = dist_2
        print()

    print(ppl, acc, d1, d2)
