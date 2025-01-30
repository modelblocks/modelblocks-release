import sys


def main():
    print("word totsurp")
    all_word, all_llmsurp, all_bori, all_bprob, all_iprob = [], [], [], [], []

    lines = sys.stdin.readlines()
    for line in lines[1:]:
        #print(line)
        word, llmsurp, bori, bprob, iprob = line.rstrip().split(" ")
        all_word.append(word)
        all_llmsurp.append(float(llmsurp))
        all_bori.append(bori)
        all_bprob.append(float(bprob))
        all_iprob.append(float(iprob))
    assert len(all_word) == len(all_llmsurp) == len(all_bori) == len(all_bprob) == len(all_iprob)

    for i in range(len(all_word)-1):
        if all_bori[i+1] == "B":
            all_llmsurp[i] -= all_bprob[i+1]
            all_llmsurp[i+1] += all_bprob[i+1]
            if all_word[i] != "<eos>":
                print(all_word[i], all_llmsurp[i])
        else:
            print(all_word[i], all_llmsurp[i])

    if all_word[-1] != "<eos>":
        print(all_word[-1], all_llmsurp[-1])


if __name__ == "__main__":
    main()
