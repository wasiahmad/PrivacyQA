from collections import Counter, OrderedDict


def reform_results(results):
    new_results = OrderedDict()
    for result in results:
        # result['id'] is company name
        # result['qid'] is the question no.
        # result['gold'] is an OrderedDict. Ex., {'Ann0' : [], 'Ann1' : [], ...}
        new_id = '{}.{}'.format(result['id'], result['qid'])
        if new_id not in new_results:
            new_results[new_id] = OrderedDict([
                ('predictions', []),
                ('gold', result['gold'])
            ])
        if result['pred'] == 1:
            new_results[new_id]['predictions'].append(result['sid'])

    return new_results


def compute_f1(gold_toks, pred_toks):
    common = Counter(gold_toks) & Counter(pred_toks)
    num_same = sum(common.values())
    if len(gold_toks) == 0 or len(pred_toks) == 0:
        # If either is no-answer, then F1 is 1 if they agree, 0 otherwise
        return [int(gold_toks == pred_toks), int(gold_toks == pred_toks), int(gold_toks == pred_toks)]
    if num_same == 0:
        return [0, 0, 0]
    precision = 1.0 * num_same / len(pred_toks)
    recall = 1.0 * num_same / len(gold_toks)
    f1 = (2 * precision * recall) / (precision + recall)
    return [precision, recall, f1]


def score(results):
    # results is a list of OrderedDict
    results = reform_results(results)
    total_scores = []

    for eid, ex in results.items():
        # loop over each query
        precision, recall, f1_score = [], [], []
        for anno_id, gold in ex['gold'].items():
            prec, rec, f1 = compute_f1(gold, ex['predictions'])
            precision.append(prec)
            recall.append(rec)
            f1_score.append(f1)

        score = OrderedDict()
        score['id'] = eid
        score['predictions'] = ex['predictions']
        score['gold'] = ex['gold']
        # We take the best
        score['precision'] = max(precision)
        score['recall'] = max(recall)
        score['f1'] = max(f1_score)
        total_scores.append(score)

    return total_scores
