import sys, os, re

def uniq(lst):
    last = object()
    for item in lst:
        if item == last:
            continue
        yield item
        last = item

def labeled_cues(graph_file):
    labeled_cues = []
    article_num = 0
    sentence_num = 0
    nodes_to_preds = {}

    with open(graph_file, "r") as cuegraphs:
        section_num = str(graph_file).replace(".casp.cuegraphs", "")[-2:]
        data = cuegraphs.readlines()
        # data = [line for line in data if line != "\n"]

    for line in data:

        # removing FAIL trees
        if re.search("FAIL", line) is not None:
            continue

        # if re.search("!article", line):
        if line == "\n":
            article_num += 1
            sentence_num = 0
            continue
        sentence_num += 1
        cuechunks = line.split()
        for chunk in cuechunks:
            cues = chunk.split(",")

            # to accommodate for ",:," which is split by cuechunk.split(",")
            if len(cues) == 5:
                del cues[3:5]
                cues[2] = ",:,"

            # removing 4+ relationships
            if cues[1] not in ["0", "1", "2", "3", "c", "e", "h", "r"]:
                continue

            # add sentence IDs to "nodes" (which start with numbers and not part-of-speech)
            for i in [0, 2]:
                if re.match("[0-9]", cues[i]):
                    cues[i] = str(section_num).zfill(2) + str(sentence_num).zfill(4) + cues[i]

            # for removing 0 relationship later
            if cues[1] == "0":
                nodes_to_preds[cues[0]] = cues[2]
                continue

            cues.append(str(section_num).zfill(2) + str(sentence_num).zfill(4))
            labeled_cues.append(cues)

    # returns nested list
    return labeled_cues, nodes_to_preds

def remove_inheritance(cue_list, cue_dict):
    cleaned_cues = []
    for cue in cue_list:
        if cue[1] in ["c", "e", "h", "r"]:
            if cue[2] in cue_dict:
                print(cue)
                for rel in cue_dict[cue[2]]:
                    # prevent inheriting 0 relationships that go to predicates
                    if rel == "0":
                        continue
                    cleaned_cues.append([cue[0], rel, cue_dict[cue[2]][rel], cue[3]])
            continue
        cleaned_cues.append(cue)
    return cleaned_cues

def cleaned_cues(labeled_cues, nodes_to_preds):
    # initialize and populate nested dict
    cue_dict = {}
    for cue in labeled_cues:
        cue_dict[cue[0]] = cue_dict.get(cue[0], {})
        cue_dict[cue[0]][cue[1]] = cue[2]

    # take care of dependencies
    cleaned_cues = remove_inheritance(labeled_cues, cue_dict)
    patience = 0
    while cleaned_cues != remove_inheritance(cleaned_cues, cue_dict):
        cleaned_cues = remove_inheritance(cleaned_cues, cue_dict)
        patience += 1
        if patience == 100:
            for cue in cleaned_cues:
                if cue[1] in ["c", "e", "h", "r"]:
                    cleaned_cues.remove(cue)
            print("Circular cues found")
            break

    uniq_cleaned_cues = list(uniq(sorted(cleaned_cues)))

    # substitute "-0" nodes with predicates
    for cue in uniq_cleaned_cues:
        for i in [0, 2]:
            if cue[i] in nodes_to_preds:
                cue[i] = nodes_to_preds[cue[i]]

    return uniq_cleaned_cues

def main(graph_dir, target_dir):
    # current version of RGCN code does not need dev/test data
    filelist = []
    cues = []
    sents = []
    entities = []
    relations = []
    # dev_cues = []

    for file in os.listdir(graph_dir):
        if file.endswith("cuegraphs"):
            filelist.append(file)

    for file in sorted(filelist):
        label_cues, nodes_to_preds = labeled_cues(os.path.join(graph_dir, file))
        clean_cues = cleaned_cues(label_cues, nodes_to_preds)
        cues.extend(clean_cues)
        # if file == sorted(filelist)[-1]:
        #     dev_cues.extend(clean_cues)

    os.makedirs(os.path.dirname(str(target_dir) + "/train.txt"), exist_ok=True)
    with open(target_dir + "/train.txt", "w") as f:
        for cue in cues:
            entities.extend([cue[0], cue[2]])
            relations.append(cue[1])
            sents.append(cue[3])
            # getting rid of "negative" relationships for now...?
            # relations.append("-" + str(cue[1]))
            f.write(str(cue[0]) + "\t" + str(cue[1]) + "\t" + str(cue[2]) + "\t" + str(cue[3]) + "\n")
            # f.write(str(cue[2]) + "\t" + "-" + str(cue[1]) + "\t" + str(cue[0]) + "\n")

    # os.makedirs(os.path.dirname(str(target_dir) + "/valid.txt"), exist_ok=True)
    # with open(target_dir + "/valid.txt", "w") as f:
    #     for cue in dev_cues:
    #         entities.extend([cue[0], cue[2]])
    #         relations.append(cue[1])
    #         relations.append("-" + str(cue[1]))
    #         f.write(str(cue[0]) + "\t" + str(cue[1]) + "\t" + str(cue[2]) + "\n")
    #         f.write(str(cue[2]) + "\t" + "-" + str(cue[1]) + "\t" + str(cue[0]) + "\n")
    #
    # os.makedirs(os.path.dirname(str(target_dir) + "/test.txt"), exist_ok=True)
    # with open(target_dir + "/test.txt", "w") as f:
    #     for cue in dev_cues:
    #         entities.extend([cue[0], cue[2]])
    #         relations.append(cue[1])
    #         relations.append("-" + str(cue[1]))
    #         f.write(str(cue[0]) + "\t" + str(cue[1]) + "\t" + str(cue[2]) + "\n")
    #         f.write(str(cue[2]) + "\t" + "-" + str(cue[1]) + "\t" + str(cue[0]) + "\n")

    entities = list(enumerate(sorted(set(entities))))
    relations = list(enumerate(sorted(set(relations))))
    sents = list(enumerate(sorted(set(sents))))

    os.makedirs(os.path.dirname(str(target_dir) + "/entities.dict"), exist_ok=True)
    with open(str(target_dir) + "/entities.dict", "w") as file:
        for k in entities:
            file.write(str(k[0]) + "\t" + str(k[1]) + "\n")

    os.makedirs(os.path.dirname(str(target_dir) + "/relations.dict"), exist_ok=True)
    with open(str(target_dir) + "/relations.dict", "w") as file:
        for k in relations:
            file.write(str(k[0]) + "\t" + str(k[1]) + "\n")

    os.makedirs(os.path.dirname(str(target_dir) + "/sentences.dict"), exist_ok=True)
    with open(str(target_dir) + "/sentences.dict", "w") as file:
        for k in sents:
            file.write(str(k[0]) + "\t" + str(k[1]) + "\n")

if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])