import random
from typing import List


def generate_samples(
        content: List[str],
        num_samples: int,
        min_sentences_in_sample: int,
        max_sentences_in_sample: int,
        min_sequence_len: int,
        max_sequence_len: int,
        max_tries: int = 100
) -> List[str]:
    used_start_pos = set()
    result = list()
    tries_count = 0

    while len(result) < num_samples and tries_count < max_tries:
        tries_count += 1

        start_pos = random.randrange(0, len(content))

        if start_pos in used_start_pos:
            continue

        num_sentences = random.randint(min_sentences_in_sample, max_sentences_in_sample)

        if start_pos + num_sentences - 1 >= len(content):
            continue

        sentences = [content[start_pos + i] for i in range(num_sentences)]
        sample = " ".join(sentences)
        if min_sequence_len <= len(sample) <= max_sequence_len:
            result.append(sample)
            used_start_pos.add(start_pos)
            tries_count = 0

    return result
