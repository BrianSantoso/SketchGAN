# SketchGAN
Deep Convolutional Generative Adversarial Network (DCGAN) for generating pencil sketches

Walking Through Latent Space:


<img src="https://raw.githubusercontent.com/BrianSantoso/SketchGAN/master/samples/latent2.gif" width="256">


(Linear interpolation between latent vectors)


The model contains a total of 24,216,196 trainable parameters.

The final layer of the generator network utilizes a sigmoid function that is stretched by a factor of 5 (by scaling the input by 0.2). This allows the model to more easily learn smooth pencil-like textures (as opposed to crisp, pen-like figures like the digits from the MNIST dataset).

The training set consists of 1001 unique grayscale images of pencil sketches of birds, horses, butterflies, disney characters, male and female faces, flowers, hearts, hands, still life, and other miscellanious objects. The image data was also flipped across the vertical axis to produce a total of 2002 pencil sketches.



<img src="https://github.com/BrianSantoso/SketchGAN/blob/master/samples/363000_7.PNG" width="614">
(generated sketches after 363k training iterations)

<img src="https://github.com/BrianSantoso/SketchGAN/blob/master/samples/363000_6.PNG" width="614">
<img src="https://github.com/BrianSantoso/SketchGAN/blob/master/samples/363000_1.PNG" width="614">
<img src="https://github.com/BrianSantoso/SketchGAN/blob/master/samples/363000_2.PNG" width="614">

Miscellanious sketches at random training iterations:


<img src="https://raw.githubusercontent.com/BrianSantoso/SketchGAN/master/samples/imgif1.jpg" width="128">
<img src="https://github.com/BrianSantoso/SketchGAN/blob/master/samples/imgif837.jpg" width="128">
<img src="https://github.com/BrianSantoso/SketchGAN/blob/master/samples/imgif225.jpg" width="128">
<img src="https://github.com/BrianSantoso/SketchGAN/blob/master/samples/499500_3.PNG" width="614">
<img src="https://github.com/BrianSantoso/SketchGAN/blob/master/samples/499500_1.PNG" width="614">
<img src="https://github.com/BrianSantoso/SketchGAN/blob/master/samples/499500_2.PNG" width="614">
<img src="https://github.com/BrianSantoso/SketchGAN/blob/master/samples/299500_1.PNG" width="614">
<img src="https://github.com/BrianSantoso/SketchGAN/blob/master/samples/299500_2.PNG" width="614">
<img src="https://github.com/BrianSantoso/SketchGAN/blob/master/samples/499500_4.PNG" width="614">
<img src="https://github.com/BrianSantoso/SketchGAN/blob/master/samples/499500_5.PNG" width="614">
<img src="https://github.com/BrianSantoso/SketchGAN/blob/master/samples/499500_6.PNG" width="614">
<img src="https://github.com/BrianSantoso/SketchGAN/blob/master/samples/499500_7.PNG" width="614">
<img src="https://github.com/BrianSantoso/SketchGAN/blob/master/samples/499500_8.PNG" width="614">
<img src="https://github.com/BrianSantoso/SketchGAN/blob/master/samples/499500_9.PNG" width="614">
<img src="https://github.com/BrianSantoso/SketchGAN/blob/master/samples/499500_10.PNG" width="614">
<img src="https://github.com/BrianSantoso/SketchGAN/blob/master/samples/499500_11.PNG" width="614">
<img src="https://github.com/BrianSantoso/SketchGAN/blob/master/samples/499500_12.PNG" width="614">
<img src="https://github.com/BrianSantoso/SketchGAN/blob/master/samples/363000_1.PNG" width="614">
<img src="https://github.com/BrianSantoso/SketchGAN/blob/master/samples/363000_2.PNG" width="614">
<img src="https://github.com/BrianSantoso/SketchGAN/blob/master/samples/363000_3.PNG" width="614">
<img src="https://github.com/BrianSantoso/SketchGAN/blob/master/samples/363000_4.PNG" width="614">
<img src="https://github.com/BrianSantoso/SketchGAN/blob/master/samples/363000_5.PNG" width="614">
<img src="https://github.com/BrianSantoso/SketchGAN/blob/master/samples/363000_6.PNG" width="614">
<img src="https://github.com/BrianSantoso/SketchGAN/blob/master/samples/363000_7.PNG" width="614">
<img src="https://github.com/BrianSantoso/SketchGAN/blob/master/samples/363000_8.PNG" width="614">
<img src="https://github.com/BrianSantoso/SketchGAN/blob/master/samples/79500_3.PNG" width="614">
<img src="https://github.com/BrianSantoso/SketchGAN/blob/master/samples/100000_3.PNG" width="614">
<img src="https://github.com/BrianSantoso/SketchGAN/blob/master/samples/100000_4.PNG" width="614">
<img src="https://github.com/BrianSantoso/SketchGAN/blob/master/samples/100000_5.PNG" width="614">
