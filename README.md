# SketchGAN
Deep Convolutional Generative Adversarial Network (DCGAN) for generating pencil sketches


Generated sketches after 363k training iterations

<img src="https://github.com/BrianSantoso/SketchGAN/blob/master/samples/363000_7.PNG" width="614">
<img src="https://github.com/BrianSantoso/SketchGAN/blob/master/samples/363000_6.PNG" width="614">
<img src="https://github.com/BrianSantoso/SketchGAN/blob/master/samples/363000_1.PNG" width="614">
<img src="https://github.com/BrianSantoso/SketchGAN/blob/master/samples/363000_2.PNG" width="614">

The model contains a total of 24,216,196 trainable parameters.

The final layer of the generator network utilizes a sigmoid function that is stretched by a factor of 5 (by scaling the input by 0.2). This allows for smooth pencil-like textures (as opposed to crisp, pen-like figures like the digits from the MNIST dataset).

The training set consists of 1001 unique grayscale images of pencil sketches of birds, horses, butterflies, disney characters, male and female faces, flowers, hearts, hands, still life, and other miscellanious objects. The image data was also flipped across the vertical axis to produce a total of 2002 pencil sketches.



Generated sketches after 299.5k training iterations

<img src="https://github.com/BrianSantoso/SketchGAN/blob/master/samples/299500_1.PNG" width="614">
<img src="https://github.com/BrianSantoso/SketchGAN/blob/master/samples/299500_2.PNG" width="614">



Miscellanious sketches at random training iterations:

<img src="https://github.com/BrianSantoso/SketchGAN/blob/master/samples/79500_3.PNG" width="614">
<img src="https://github.com/BrianSantoso/SketchGAN/blob/master/samples/100000_3.PNG" width="614">
<img src="https://github.com/BrianSantoso/SketchGAN/blob/master/samples/100000_4.PNG" width="614">
<img src="https://github.com/BrianSantoso/SketchGAN/blob/master/samples/100000_5.PNG" width="614">
