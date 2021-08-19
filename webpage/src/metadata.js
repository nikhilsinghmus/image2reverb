const metadata = {
    img: "./splash.png",
    abstract: "Measuring the acoustic characteristics of a space is often done by capturing its impulse response (IR), a representation of how a full-range stimulus sound excites it. This work generates an IR from a single image, which can then be applied to other signals using convolution, simulating the reverberant characteristics of the space shown in the image. Recording these IRs is both time-intensive and expensive, and often infeasible for inaccessible locations. We use an end-to-end neural network architecture to generate plausible audio impulse responses from single images of acoustic environments. We evaluate our method both by comparisons to ground truth data and by human expert evaluation. We demonstrate our approach by generating plausible impulse responses from diverse settings and formats including well known places, musical halls, rooms in paintings, images from animations and computer games, synthetic environments generated from text, panoramic images, and video conference backgrounds.",
    tldr: "We present a method for generating audio impulse responses, to simulate the acoustic reverberation of a given environment, from a 2D image.",
    venue: "ICCV 2021",
    paper: "https://arxiv.org/abs/2103.14201",
    bibtex: `@inproceedings{singh2021image2reverb,
    author = {Singh, Nikhil and Mentch, Jeff and Ng, Jerry and Beveridge, Matthew and Drori, Iddo},
    title = {Image2Reverb: Cross-Model Reverb Impulse Response Synthesis},
    booktitle = {IEEE/CVF International Conference on Computer Vision (ICCV)},
    month = {October},
    year = {2021}
}`,
    github: "https://github.com/nikhilsinghmus/image2reverb"
};

export default metadata;