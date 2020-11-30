import sys
import numpy
import pandas
import krippendorff


def main():
    f = sys.argv[1]
    places = ["Large Hall", "Studio", "Medium Hall", "Outdoor", "Small Space", "Home Entryway", "Living Room"]
    d = pandas.read_csv(f, header=2).fillna(0)
    data = d.values.T

    q_ratings = data[18:-2:2,:].reshape(7, -1).astype(int)
    m_ratings = data[19:-1:2,:].reshape(7, -1).astype(int)

    mean_q = numpy.mean(q_ratings, axis=1)
    mean_m = numpy.mean(m_ratings, axis=1)

    categories_q = q_ratings.reshape(7, 2, -1).mean(axis=1)
    categories_m = m_ratings.reshape(7, 2, -1).mean(axis=1)

    print_table(places, mean_q, mean_m)
    print(get_agreement(categories_q), get_agreement(categories_m))


def print_table(*rows):
    for r in rows:
        r_text = ""
        if isinstance(r, numpy.ndarray):
            r = r.round(2)
        r_text = " & ".join(map(str, r))
        print(r_text)


def get_agreement(r):
    return krippendorff.alpha(r).round(3)

if __name__ == "__main__":
    main()