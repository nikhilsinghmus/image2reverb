#rough code to save processed dataset in pix2pix input format
import sys, os, glob
import itertools
from shutil import copyfile
#dataset_outdir = './'

dataset_outdir =  '/datasets/room2reverb/' #where to save dataset files
if not os.path.exists(dataset_outdir+'train_img'):
    os.mkdir(dataset_outdir+'train_img')
    os.mkdir(dataset_outdir+'train_label')

    
dataset_dir = os.path.abspath('../dataset/Room2Reverb/IR_Collection/') #input dataset
standardized_dir = '../standardized_data/'
dirs = glob.glob(dataset_dir+'/*')

for i,d in enumerate(dirs):
    wavlist = []
    wavlist = glob.glob(dirs[i]+'/*.wav') #make a list of the wav files
    imglist = []
    imglist = glob.glob(dirs[i]+'/*.jpg') #list of jpegs
    all_pairs = list(itertools.product(wavlist, imglist)) #get all pairs
    for i_inner,pair_filenames in enumerate(all_pairs):
        current_file = os.path.splitext(os.path.basename(pair_filenames[1]))[0]
        subfolder = os.path.basename(os.path.dirname(pair_filenames[1]))
        imgname = standardized_dir+subfolder+'/'+current_file+'.jpg'
        copyfile(imgname, dataset_outdir+'train_img/'+str(i)+'_'+str(i_inner)+'.jpg')
        labelname = os.path.splitext(pair_filenames[0])[0]+'.png'
        copyfile(labelname, dataset_outdir+'train_label/'+str(i)+'_'+str(i_inner)+'.png')
