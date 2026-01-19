import argparse


def evaluate_rank_metrics(dataset, model):
    rank_file_path = f'../rank_file/{dataset}/{dataset}_{model}.txt'

    reciprocal_ranks = []
    hits_at_1 = 0
    hits_at_3 = 0
    hits_at_10 = 0
    total = 0

    try:
        with open(rank_file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split('\t')
                if len(parts) != 5:
                    continue
                try:
                    rank = int(parts[4])
                except ValueError:
                    continue

                reciprocal_ranks.append(1.0 / rank)
                if rank <= 1:
                    hits_at_1 += 1
                if rank <= 3:
                    hits_at_3 += 1
                if rank <= 10:
                    hits_at_10 += 1
                total += 1

        if total == 0:
            print("There is no valid data in the file")
            return

        mrr = sum(reciprocal_ranks) / total
        h1 = hits_at_1 / total
        h3 = hits_at_3 / total
        h10 = hits_at_10 / total

        print(f"Total sample size: {total}")
        if model != 'ICL' and model != 'GenTKG':
            print(f"MRR: {mrr:.4f}")
        print(f"Hits@1: {h1:.4f}")
        print(f"Hits@3: {h3:.4f}")
        print(f"Hits@10: {h10:.4f}")

    except Exception as e:
        print(f"An error occurred while reading the file: {e}")


def evaluate_weighted_rank_metrics(dataset, model, bias):
    rank_file_path = f'../rank_file/{dataset}/{dataset}_{model}.txt'
    strikingness_file_path = f'../data/{dataset}/0.01_0.1_200_0.4_0.4_0.2/test_extend.txt'
    stat_file_path = f'../data/{dataset}/stat.txt'

    with open(stat_file_path, 'r', encoding='utf-8') as f:
        stat = f.read().strip().split('\t')
        rel_num = int(stat[1])

    strikingness_dict = {}

    try:
        with open(strikingness_file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split('\t')
                if len(parts) != 5:
                    continue
                try:
                    s, r, o, t = map(int, parts[:4])
                    strikingness = float(parts[4])
                    key = (s, r, o, t)
                    reverse_key = (o, rel_num + r, s, t)
                    strikingness_dict[key] = strikingness + bias
                    strikingness_dict[reverse_key] = strikingness + bias
                except (ValueError, IndexError):
                    continue
    except Exception as e:
        print(f"Error occurred while reading the significance file: {e}")
        return

    if not strikingness_dict:
        print("There is no valid data in the significance file.")
        return

    total_strikingness = sum(strikingness_dict.values())
    if total_strikingness == 0:
        print("The sum of significance is 0, thus the weight cannot be calculated.")
        return

    weighted_reciprocal_ranks = 0.0
    weighted_hits_at_1 = 0.0
    weighted_hits_at_3 = 0.0
    weighted_hits_at_10 = 0.0
    total_samples = 0

    try:
        with open(rank_file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split('\t')
                if len(parts) != 5:
                    continue
                try:
                    s, r, o, t = map(int, parts[:4])
                    rank = int(parts[4])
                    key = (s, r, o, t)
                    strikingness = strikingness_dict.get(key, 0.0)
                    weight = strikingness / total_strikingness

                    weighted_reciprocal_ranks += weight * (1.0 / rank)
                    if rank <= 1:
                        weighted_hits_at_1 += weight
                    if rank <= 3:
                        weighted_hits_at_3 += weight
                    if rank <= 10:
                        weighted_hits_at_10 += weight
                    total_samples += 1

                except (ValueError, IndexError):
                    continue
    except Exception as e:
        print(f"Error occurred while reading the ranking file: {e}")
        return

    if total_samples == 0:
        print("There is no valid data in the ranking file")
        return

    if model != 'ICL' and model != 'GenTKG':
        print(f"WMRR: {weighted_reciprocal_ranks:.4f}")
    print(f"WHits@1: {weighted_hits_at_1:.4f}")
    print(f"WHits@3: {weighted_hits_at_3:.4f}")
    print(f"WHits@10: {weighted_hits_at_10:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluate rank metrics')
    parser.add_argument('--dataset', '-d', default='ICEWS14', type=str, help='Path to the rank file')
    parser.add_argument('--model', default='regcn', type=str)
    parser.add_argument('--bias', default=0.0, type=float)
    args = parser.parse_args()
    dataset = args.dataset
    model = args.model
    bias = args.bias

    print(f"Dataset: {dataset}, Model: {model}, Bias: {bias}")
    evaluate_rank_metrics(dataset, model)
    evaluate_weighted_rank_metrics(dataset, model, bias)