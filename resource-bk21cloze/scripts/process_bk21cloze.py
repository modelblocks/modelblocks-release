import pandas as pd
import sys

# pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)

# interleave ITEM-5 and condition for sentid; HC MC LC
# position -> sentpos; SUM_3RT -> fdur
def main():
    df = pd.read_csv(sys.argv[1])
    df = df.rename(columns={"SUB": "subject", "critical_word": "word", "position": "sentpos", "SUM_3RT": "fdur"})
    condition_to_idx = {"HC": 0, "MC": 1, "LC": 2}
    df["condition"] = df["condition"].map(condition_to_idx)
    df["ITEM"] -= 5
    df["sentid"] = 3 * df["ITEM"] + df["condition"]

    df.to_csv(sys.stdout, sep=" ", index=False)


if __name__ == "__main__":
    main()
