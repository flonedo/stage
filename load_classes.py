# reads classes from a text file
def get_dataset_classes(synset_path):
    classes = []
    with open(synset_path) as syn:
        for line in syn:
            classes.append(line)
    return classes
