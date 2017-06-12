#Steps of understanding:
The idea here is to provide a conceptual story, and then to have the code well commented to provide some of the technical understanding

I was able to adapt the MNIST for beginners (link?) tf tutorial reasonably easily to handle my data, it was 
a matter of changing the outputs from 10 categories to a whole image vector. The idea for my network being that it would take 
as input an image encoded in just the red channel, and output an image vector of the same dimesions that would be a heatmap of locations
of intersects between wires. For training data, this heatmap is encoded in the green channel of png images in the RGBA format.

## Making the training data
Needed to ensure that the line intersections corresponded to segment intersections, and that they were on the image.
in order to make the images more realistic, the segments sometimes start/end off screen, thus the code had to catch intersects that 
existed between the segments, but off-screen.
## loading the training data
The data is loaded as lists of matrices and has functions to get batches of images as withthe MNIST example.
