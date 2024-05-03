# Simplification
The purpose of this folder is to sidestep the previous issues that I had with RLlib. The differences are as follows:

### Using a simplier framework
RLlib is excellent for some things, but horrific for rapid prototyping, testing/training/deploying models beyond itself, and living a happy life. As such, this folder will use Stable Baseline 3, which is much easier to work with then RLlib. 

### Using a simpler environment
Previously I tried starting on the much more challenging maze environment. This proved to be challenging enough that it made figuring out the origin of the various problems almost impossible, with any run that resulted in the model learning anything worthwhile taking hours to run. As such, this folder will deal with a simple gridworld that can be learned relatively quickly, as well as simple gym environments that are even easier. 

### Train the bijective part of the model seperately
By training g inverse seperately, I wont need to worry about extra loss functions. 