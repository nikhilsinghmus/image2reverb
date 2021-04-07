# Image2Reverb


Image2Reverb is an end-to-end neural network that generates plausible audio impulse responses from single images of acoustic environments. Code for the paper [Image2Reverb: Cross-Modal Reverb Impulse Response Synthesis](https://arxiv.org/abs/2103.14201). The architecture is a conditional GAN with a ResNet50 (pre-trained on Places365 and fine-tuned) image encoder. It generates monoaural audio impulse responses (directly applicable to convolution applications) as magnitude spectrograms.

## Dependencies

**Model/Data:**

* PyTorch>=1.7.0
* PyTorch Lightning
* torchvision
* torchaudio
* librosa
* PyRoomAcoustics
* PIL

**Eval/Preprocessing:**

* PySoundfile
* SciPy
* Scikit-Learn
* python-acoustics
* google-images-download
* matplotlib


## Usage

We will make a pre-trained model available soon!

## Acknowledgments

We borrow and adapt code snippets from [GANSynth](https://github.com/magenta/magenta/tree/master/magenta/models/gansynth) (and [this](https://github.com/ss12f32v/GANsynth-pytorch) PyTorch re-implementation), additional snippets from [this](https://github.com/shanexn/pytorch-pggan) PGGAN implementation, [monodepth2](https://github.com/nianticlabs/monodepth2), [this](https://github.com/jacobgil/pytorch-grad-cam) GradCAM implementation, and more.
