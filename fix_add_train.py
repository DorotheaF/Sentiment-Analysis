import csv

with open("training-data/train.csv", newline='', encoding="utf8") as csvfile:
    reader = csv.reader(csvfile, delimiter=',', quotechar='"')
    with open("training-data/additional-train.txt", "w", encoding="utf8") as writer:
        i = 1
        for review in reader:
            if review[0] != "#NAME?":
                if review[1] == "1":
                    cat = 1
                else:
                    cat = 0
            rev = review[0].replace("\n", " ")
            writer.write(str(cat) + " " + "ID-" + str(i) + " " + rev + "\n")
            i = i+1



