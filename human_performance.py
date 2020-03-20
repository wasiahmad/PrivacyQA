import os
import numpy as np
from collections import Counter


def get_company_names(src_dir):
    filenames = os.listdir(src_dir)
    com_names = set()
    for f in filenames:
        com_names.add(os.path.splitext(f)[0])

    return list(com_names)


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


def read_labels(filename):
    labels = list()
    with open(filename) as f:
        for line in f:
            tokens = line.strip().split('\t')
            assert len(tokens) >= 2
            if len(tokens) == 2:
                labels.append({0: []})
            else:
                one_label = {}
                for anno_id, anno in enumerate(tokens[2:]):
                    if anno == '_':
                        one_label[anno_id] = []
                    else:
                        one_label[anno_id] = [int(v) for v in anno.split(',')]
                assert len(one_label) >= 1
                labels.append(one_label)

    return labels


def main(src_dir):
    company_names = get_company_names(src_dir)

    full_test_score = np.array([0.0, 0.0, 0.0])
    full_total = 0
    for cname in company_names:
        labels = read_labels(os.path.join(src_dir + '%s.label' % cname))
        print(cname)
        for idx, one_label in enumerate(labels):
            query_score = np.array([0.0, 0.0, 0.0])
            query_total = 0
            print(one_label)
            for j in range(len(one_label)):
                other_ann = []
                for k in range(len(one_label)):
                    if j != k:
                        other_ann.append(one_label[k])

                all_reference_annotations = [
                    compute_f1(o_annotation, one_label[j]) for
                    o_annotation in other_ann]

                # We take the best
                best = max(all_reference_annotations, key=lambda x: x[2])

                query_score += best
                query_total += 1

            if query_total > 0:
                per_query_score = query_score / query_total
                full_test_score += per_query_score
                full_total += 1

    precision, recall = full_test_score[0] / full_total, full_test_score[1] / full_total
    f1 = (2 * precision * recall) / (precision + recall)
    print('[ precision = {:.2f} | recall = {:.2f} | f1 = {:.2f} | examples = {} ]'.format(
        precision * 100, recall * 100, f1 * 100, full_total))


if __name__ == '__main__':
    main('./data/privacyQA/test/')
