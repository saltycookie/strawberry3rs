import random
from collections import defaultdict


def create_word_occurrence_dictionary(words):
    """
    Generates a dictionary of dictionaries representing word occurrences.

    Given a list of words, this function creates a dictionary where:
    - The top-level keys are single letters (characters) found in the words.
    - The second-level keys are the number of times that letter occurs in a word.
    - The values associated with the second-level keys are *randomly shuffled*
    lists of words from the input list that have the corresponding letter
    occurring the specified number of times.

    Args:
        words: A list of strings (words).

    Returns:
        A dictionary of dictionaries structured as described above.
    """

    d = defaultdict(lambda: defaultdict(list))

    for word in words:
        letter_counts = {}
        for letter in word:
            letter_counts[letter] = letter_counts.get(letter, 0) + 1

        for letter, count in letter_counts.items():
            d[letter][count].append(word)

    # Shuffle the lists of words for each letter and occurrence count
    for letter in d:
        for count in d[letter]:
            random.shuffle(d[letter][count])

    return d


def create_dataset(occurrence_dict, max_occurrence, num_words_per_letter_occur_pair):
    examples = []
    used = {}
    for letter in "abcdefghijklmnopqrstuvwxyz":
        for occur in range(1, max_occurrence + 1):
            for _ in range(num_words_per_letter_occur_pair):
                for word in occurrence_dict[letter][occur]:
                    if not used.get(word):
                        used[word] = True
                        examples.append((word, letter, occur))
                        break
                else:
                    for reduced_occur in range(occur - 1, 0, -1):
                        found = False
                        for word in occurrence_dict[letter][reduced_occur]:
                            if not used.get(word):
                                used[word] = True
                                made_up_word = word + letter * (occur - reduced_occur)
                                examples.append((made_up_word, letter, occur))
                                found = True
                                break
                        if found:
                            break
                    else:
                        print(
                            f"I couldn't create a word with letter {letter} occurring {occur} number of times."
                        )
    return examples


if __name__ == "__main__":
    import nltk
    from sklearn.model_selection import train_test_split
    from datasets import Dataset, DatasetDict

    nltk.download("words")
    all_words = nltk.corpus.words.words()
    wo_dict = create_word_occurrence_dictionary(all_words)
    examples = create_dataset(wo_dict, 5, 10)
    qa_pairs = []
    for example in examples:
        question = f"How many {example[1]}'s are in the word {example[0]}?"
        qa_pairs.append({"question": question, "answer": example[2]})

    train_data, test_data = train_test_split(qa_pairs, train_size=0.8, random_state=97)
    train_dataset = Dataset.from_dict(
        {
            "question": [qa["question"] for qa in train_data],
            "answer": [qa["answer"] for qa in train_data],
        }
    )
    test_dataset = Dataset.from_dict(
        {
            "question": [qa["question"] for qa in test_data],
            "answer": [qa["answer"] for qa in test_data],
        }
    )
    dataset = DatasetDict({"train": train_dataset, "test": test_dataset})
    # Print some examples
    print("Training Examples:")
    for i in range(10):
        print(dataset["train"][i])

    print("\nTest Examples:")
    for i in range(10):
        print(dataset["test"][i])
    print(dataset)
    dataset.save_to_disk("character_count_dataset")
