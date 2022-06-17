import sys, re, pandas as pd

docs = {
    "Stories_1_boar_T": "Boar",
    "Stories_2_aqua_T": "Aqua",
    "Stories_3_matchstickseller_N": "MatchstickSeller",
    "Stories_4_kingofbirds_N": "KingOfBirds",
    "Stories_5_elvis_T": "Elvis",
    "Stories_6_mrsticky_N": "MrSticky",
    "Stories_7_highschool_N": "HighSchool",
    "Stories_9_tulips_T": "Tulips",
    "StoriesAud_1_boar_T": "Boar",
    "StoriesAud_2_aqua_T": "Aqua",
    "StoriesAud_4_kingofbirds_N": "KingOfBirds",
    "StoriesAud_5_elvis_T": "Elvis",
    "Stories_ToM_1_boar_T": "Boar",
    "Stories_ToM_2_aqua_T": "Aqua",
    "Stories_ToM_4_kingofbirds_N": "KingOfBirds",
    "Stories_ToM_5_elvis_T": "Elvis",
    "Stories_ToM_7_highschool_N": "HighSchool"
}

df = pd.read_csv(sys.stdin)

df = df[df.Story.isin(docs.keys())]

#patt = re.compile("([A-Za-z]+)_.$")

# Simplify each story name
# e.g. "Stories_1_boar_T" -> "Boar"
concise_story = list()
for _, row in df.iterrows():
    #concise_story.append(patt.search(row.Story).group(1).capitalize())
    concise_story.append(docs[row.Story])
df.Story = concise_story

#d.to_csv(sys.stdout, index=False, na_rep='nan')
df.to_csv(sys.stdout, index=False)
