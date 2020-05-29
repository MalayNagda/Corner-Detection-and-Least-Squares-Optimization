# Corner Detection and Least Squares Optimization

The objective was to create a 300x300 square with a white irregular quadrilateral in it, rotate and translate the image about the original center and detect corners in the two images.
After this, the rotation and translation in the second image was to be recovered so that one can define the original image again using the the transformed second image using least squares optimization.

Detected corners in the original image are-

<p align="center">
  <img src="images/Detected_corners_1.png">
</p>

Detected corners in the transformed image are-

<p align="center">
  <img src="images/Detected_corners_2.png">
</p>

The position of the corners detected in the original image are-
| 133.84070397 | 176.2096639 |
| ------------- | ------------- |
| **131.0139748** | **108.62686005** |
| **171.69805636** | **169.8760343** |
| **177.44726487** | **129.28744175** |

The recovered position from the corners detected in the transformed image are-
| 125 | 179 |
| ------------- | ------------- |
| **136** | **115** |
| **169** | **165** |
| **184** | **125** |
