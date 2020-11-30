import sys
from analyze_data import print_table


def main():
    categories_out = sys.argv[1]
    places = ["Large Hall", "Studio", "Medium Hall", "Outdoor", "Small Space", "Home Entryway", "Living Room"]
    for i, place in enumerate(places):
        print(i, place)

    with open(categories_out) as infile:
        c = infile.read().split("\n")
        c = [l.split() for l in c]

    examples = {}
    for i, (place, count) in enumerate(c):
        category = input("%s: " % place)
        examples.setdefault(category, 0)
        examples[category] += int(count)
        print ("\033[A                             \033[A")
        print(examples)

    print_table([places[int(i)] for i in examples.keys()], list(examples.values()))



if __name__ == "__main__":
    main()