# unpaired-image-to-image-translation-cyclegans
Unpaired Image-to-Image Translation using CycleGANs (horse2zebra)

# This Repository Contains Two Files:
## Code
Developed a 70x70 PatchGAN Discriminator and a 24-layer Generator model for effective unpaired image translation tasks. The model was trained using the horse2zebra dataset, running for 5 epochs to generate realistic fake images of horses transformed into zebras. This approach successfully reduced both discriminator and generator losses, demonstrating the effectiveness of the developed models in handling unpaired image translation tasks.

## Dataset
The dataset is available at UC Berkeley. You can go ahead and access it here: https://efrosgans.eecs.berkeley.edu/cyclegan/datasets/.

## Instance Normalization (.py) File
Contains code for instance normalization used in both the Discriminator (70x70 PatchGAN) and Generator models.

## References
https://github.com/bnsreenu/python_for_microscopists/tree/master/253_254_cycleGAN_monet2photo
https://arxiv.org/abs/1703.10593
