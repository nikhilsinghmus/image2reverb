import sys
import numpy
import soundfile
import librosa
from PIL import Image

def main():
    export_format = 'png'
    #import_format = 'npy'
    for f in sys.argv[1:]:
        y, fs = soundfile.read(f)
        spec = librosa.stft(y,n_fft=1024,hop_length=256)
        spec = librosa.amplitude_to_db(abs(spec)) #because we need to do log or something to have things show up on the img
        spec = spec[0:-1,:] #get rid of nyquist frequency
        
        if export_format == 'npy':
            out_f = os.path.splitext(f)[0]+'.npy'
            #numpy.save(out_f,spec)
        else: 
            spec= spec - spec.min()
            spec *= 255.0/spec.max() #scale to 0 to 255
            out_f = os.path.splitext(f)[0]+'.png'
            im = Image.fromarray(spec)
            im.save(out_f)
        print("Processed and wrote %s." % out_f)

if __name__ == "__main__":
    main()
