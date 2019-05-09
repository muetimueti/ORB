# ORBextractor

A small tool that detects keypoints in an image or a sequence of images via ORB*. Meant to plug into https://github.com/raulmur/ORB_SLAM2, but works as a keypoint visualizer on its own.
The focus is on distributing the keypoints, 8 different distributions are implemented, some good, some not so much. Most relevant parameters can be adjusted via a simple gui.

*see: http://www.willowgarage.com/sites/default/files/orb_final.pdf


To analyze any single image pass: <path to settings file> <path to image> 0 (the 0 is for single image mode), eg:
```bash
ORBextractor ORBextractor/settings.yaml /home/myname/Downloads/pics/mypic.jpg 0
```
<a href="https://user-images.githubusercontent.com/27887425/57463641-76221600-727b-11e9-8af3-2334534ac622.png" target="_blank"><img src="https://user-images.githubusercontent.com/27887425/57463641-76221600-727b-11e9-8af3-2334534ac622.png" alt="ORBextractor" height="240" border="5" /></a>
  
Since this was meant for testing purposes with TUM datasets the sequence mode expects the images to be in /rgb within the folder passed via argument:
```bash
ORBextractor ORBextractor/settings.yaml /home/myname/Downloads/pics/mypic.jpg 0
```


<a href="https://user-images.githubusercontent.com/27887425/57463644-77ebd980-727b-11e9-9040-7a1b2cf2074c.png" target="_blank"><img src="https://user-images.githubusercontent.com/27887425/57463644-77ebd980-727b-11e9-9040-7a1b2cf2074c.png" alt="ORBextractor" height="240" border="5" /></a>                   


## Dependencies
### Pangolin
https://github.com/stevenlovegrove/Pangolin
### OpenCV
http://opencv.org/
