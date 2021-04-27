# Videos of the B-VAE OOD detection for different Test scenes

1. **Nom scene** - An indistribution scene in which all the feature values remain within the training distributions.

2. **HP scene** - A scene where a percipitation value outside the training distribution is introduced into the scene. In this scene the martingale of the B-VAE detector and the martingale of the precipitation reasoner increases once high precipitation is introduced. 

3. **HB scene** - A scene where a brightness value outside the training distribution is introduced into the scene. In this scene the martingale of the B-VAE detector and the martingale of the brightness reasoner increases once high brightness is introduced. 

4. **HPB scene** - A scene where both percipitation and brightness values outside the training distribution is introduced into the scene. In this scene the martingale of the B-VAE detector and the martingales of the precipitation and brightness reasoners increase once high feature values are introduced. 

5. **NR scene** - A scene with a new road segment that was not in the training distribution. In this scene, the martingale of the B-VAE detector instantaneously increases indicating the scene to be OOD. However, for most parts of the scene, the reasoners for precipitation and brightness remain low. 

