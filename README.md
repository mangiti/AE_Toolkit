# AutoEncoder Toolkit

## About
Demonstrations of AutoEncoder architecture (and variations of) and their uses


## What are AutoEncoders?

To encode is to transform data into a 'coded' representation. A 'code' may be
more secure, such that the process of decoding it is difficult or impossible 
without some hidden key or cypher, or it may simply be a more compressed and 
smaller representation of the original data. AutoEncoders represent that latter.

AutoEncoders are special networks comprised of two parts, an encoder and decoder. The encoder
takes some data $D$ that takes up some size $|D|$, and transforms it
to some latent representation $R$ such that $|R| < |D|$. The decoder then takes the latent
representation $R$ and attempts to *reconstruct* the original data $D$. To train such a network
we would minimize the reconstruction loss, usually dependent on the type of data.

### Use Cases

#### Dimension Reduction
A well-trained AutoEncoder can reduce the dimensionality of a dataset drastically while still maintaining
the most important information. In practical examples, you can reduce an image dataset from its raw pixel information
down to a lower dimensional representation such as 2–5 dimensions, and run K-Means clustering (or any other grouping algorithm)
to identify different classes.

#### Transfer Learning
The encoder portion of an AutoEncoder can also be used to conduct transfer Learning, passing the encoded output forward to a new 
network to conduct new tasks using pre-learned relationships.


### Variations

#### Variational AutoEncoders
AutoEncoders can also be used for generative purposes. If you were to pass in random noise to the decoder, it would in theory
attempt to decode it into some interpretation of the training data. The issue is that the scale and distribution of the latent
space is a mystery. Picking suitable noise for the decoder is difficult, so instead we train the latent space itself.

A Variational AutoEncoder (VAE) is very similar to a typical AutoEncoder. However rather than producing a single latent representation 
of 1xN dimensions, we create two. We then reparameterize these two such that they represent the mean and variation of a probability 
distribution. By doing this, we can ensure the latent space comes to resemble a normal distribution. To generate a new convincing
sample is as easy as drawing samples from a normal distribution.

#### Conditional Variational AutoEncoders
A new issue arises in VAEs. Say for instance that the training data consists of multiple distinct classes. A VAE would not be
able to inherently separate these classes. This means there would be no control of the class of generated samples.
It would also mean that you can't guarantee the distribution of the classes across the latent space would be optimal. 
The area between classes would be an interpolation of the two. One class may dominate the majority and thus skew all samples.

A solution is to insert some representation of class data or other meta-values of the original data into the encoder and decoder. 
This is non-latent information. For example in portrait images the meta-values could represent apparent age of an individual, the
emotion on their face, or color of their eyes. It could also be as simple as broad classifications. We call these meta-values 'conditions',
thus making this a Conditional Variational AutoEncoder.
