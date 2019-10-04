import sys, os, re

def labeled_cues(graph_file):
    labeled_cues = []
    article_num = 0
    sentence_num = 0

    with open(graph_file, "r") as cuegraphs:
        section_num = str(graph_file).replace(".casp.cuegraphs", "")[-2:]
        data = cuegraphs.readlines()
        # data = [line for line in data if line != "\n"]

    for line in data:
        # if re.search("!article", line):
        if line == "\n":
            article_num += 1
            sentence_num = 0
            continue
        sentence_num += 1
        cuechunks = line.split()
        for chunk in cuechunks:
            cues = chunk.split(",")
            for i in [0, 2]:
                # only add numbers for "nodes" (which start with numbers and not part-of-speech)
                if re.match("[0-9]", cues[i]):
                    cues[i] = str(section_num).zfill(2) + str(sentence_num).zfill(4) + cues[i]
                    # cues[i] = str(section_num).zfill(2) + str(article_num).zfill(2) + str(sentence_num).zfill(3) + cues[i]
                    # cues[i] = str(sentence_num).zfill(2) + cues[i]
            labeled_cues.append(cues)
    # returns nested list
    return labeled_cues

def remove_inheritance(cue_list, cue_dict):
    cleaned_cues = []
    for cue in cue_list:
        if cue[1] in ["c", "e", "h", "r"]:
            if cue[2] in cue_dict:
                print(cue)
                for rel in cue_dict[cue[2]]:
                    cleaned_cues.append([cue[0], rel, cue_dict[cue[2]][rel]])
            continue
        cleaned_cues.append(cue)
    return cleaned_cues

def cleaned_cues(labeled_cues):
    # initialize and populate nested dict
    cue_dict = {}
    for cue in labeled_cues:
        # to accommodate for ",:," which is split by cuechunk.split(",")
        if len(cue) == 5:
            del cue[3:5]
            cue[2] = ",:,"
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
            break
    return cleaned_cues

def main(graph_dir, target_dir):
    # current version of RGCN code does not need dev/test data
    filelist = []
    cues = []
    entities = []
    relations = []
    # dev_cues = []

    for file in os.listdir(graph_dir):
        if file.endswith("cuegraphs"):
            filelist.append(file)

    for file in sorted(filelist):
        label_cues = labeled_cues(os.path.join(graph_dir, file))
        clean_cues = cleaned_cues(label_cues)
        cues.extend(clean_cues)
        # if file == sorted(filelist)[-1]:
        #     dev_cues.extend(clean_cues)

    os.makedirs(os.path.dirname(str(target_dir) + "/train.txt"), exist_ok=True)
    with open(target_dir+"/train.txt", "w") as f:
        for cue in cues:
            entities.extend([cue[0], cue[2]])
            relations.append(cue[1])
            relations.append("-" + str(cue[1]))
            f.write(str(cue[0]) + "\t" + str(cue[1]) + "\t" + str(cue[2]) + "\n")
            f.write(str(cue[2]) + "\t" + "-" + str(cue[1]) + "\t" + str(cue[0]) + "\n")

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

    os.makedirs(os.path.dirname(str(target_dir) + "/entities.dict"), exist_ok=True)
    with open(str(target_dir) + "/entities.dict", "w") as file:
        for k in entities:
            file.write(str(k[0]) + "\t" + str(k[1]) + "\n")

    os.makedirs(os.path.dirname(str(target_dir) + "/relations.dict"), exist_ok=True)
    with open(str(target_dir) + "/relations.dict", "w") as file:
        for k in relations:
            file.write(str(k[0]) + "\t" + str(k[1]) + "\n")

if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])