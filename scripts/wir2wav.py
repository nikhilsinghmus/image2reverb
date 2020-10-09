import os
import glob
import xml.etree.ElementTree as ET
import numpy
import soundfile


def main():
    for f in glob.glob("*/**/*.xps"):
        t = ET.parse(f)
        r = t.getroot()
        d = recurse_tree(r)
        if not d:
            continue
        dirname = os.path.dirname(f)
        irname = d["IRFileNameFull"]
        path = os.path.join(dirname, irname)
        newpath = os.path.join("..", d["IRtitle"])
        if not os.path.isdir(newpath):
            os.makedirs(newpath)
        if not os.path.isfile(path):
            continue
        y = numpy.fromfile(path, numpy.float32)
        y /= float(d.get("Norm", numpy.max(y)))
        soundfile.write(os.path.join(newpath, os.path.splitext(irname)[0] + ".wav"), y, int(d["SampleRate"]), endian="little")
        print("Wrote %s to .wav." % irname)


def recurse_tree(root):
    if root.tag == "PluginSpecific":
        j = {x.attrib["Name"]:x.text for x in root[0]}
        if j.get("NumInChannels", "") == "1" and j.get("NumOutChannels", "") == "1":
            return j
    for c in root:
        d = recurse_tree(c)
        if d:
            return d


if __name__ == "__main__":
    main()