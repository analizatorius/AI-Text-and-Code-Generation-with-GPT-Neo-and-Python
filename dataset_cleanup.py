import pandas as pd

#headers
headers = ["Name", "Status", "Description", "References", "Phase", "Votes", "Comments"]

df = pd.read_csv("allitems_np.csv", skiprows = 11, sep = ',', header= 0, encoding = 'latin1', error_bad_lines=False, names=headers)

# Invalid info to drop
invalid_descriptions = ["** RESERVED **", "** REJECT **",]

maybe_ivalid = ["** DISPUTED **",]

for invalid_description in invalid_descriptions:
    df = df[df.Description.str.startswith(invalid_description, na=False) == False]
    
#Remove numbering from start of description
# df["Description"] = df.Description.replace(r'^(\d+)', r'', regex=True)



# print(df["Description"])

descriptions_to_csv = df["Description"].to_csv("descriptions.csv", index=False, sep='\t')

descriptions_to_csv = df["Description"].to_csv("descriptions_2.csv")
