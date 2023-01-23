---
title:  "Watching birds on the beach"
mathjax: true
layout: post
categories: media
---

![Swiss Alps](https://user-images.githubusercontent.com/4943215/55412536-edbba180-5567-11e9-9c70-6d33bca3f8ed.jpg)


## MathJax

All of a sudden there's a lot of birds on the beach I live in front of.  Naturally, questions arise about the seagulls:

How many birds are there? Are they always on the beach? Are they having a good time? 

These are definitely questions that can't go unanswered. Join me on a pointless journey into computer vision, machine learning and using the latest technologies to answer questions that definitely don't need to be answered. Why? Why not!

<B>THE PROBLEM</B>:

To start to address some of these burning questions, we first need a consistent view of the birds and their environment. If you're like me and have and have far too much tech lying around, one of the ESP32 microcontrollers seems like a great fit for the task. We can then add sensors to collect data to our hearts content.

![ESP32](assets/esp32.jpeg)

<b>EXPLORATIONS USING OPENCV</b>:

Rather that racing straight to using a supervised learning solution - I thought it would be fun to first explore how far we can get with computer vision.


An example image:

![EXAMPLE](/assets/OriginalImage.jpg)

There's quite a lot going on here. We'll need to do some preprocessing to get things in a suitable state for some OpenCV magic.

Given Seagulls are black and white, I think we can convert the image to greyscale. This will reduce the amount of data we need to process and some noise in the image. Helpfully, Seagulls are also black and white. The list of positive things I can say about seagulls grows! This will also be an important step later in preprocessing. 


{% highlight python %}
import cv2

def convert_to_grayscale(image):
        grey_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return grey_image

{% endhighlight %}

Looking into how we can better recognize Seagulls using the OpenCV library requires giving some thought about the shape of seagulls. Given I don't really want to spend too long thinking about seagulls, I decided to go with the most obvious thing based on the data to date.

Seagulls are usually wider than they are tall:

![SEAGULLPROOF](/assets/SeagullProof.jpg)

If we can process the image down to the essence of seagulls - we should be able to identify those shapes in the image, and count them accordingly. Granted this will only work when the seagulls are facing a certain way, but let's see how far we get and go from there!

{% highlight python %}

 cv2.connectedComponentsWithStats() 

{% endhighlight %}
A connected component is basically a blob of pixels that are joined together. 

We should be able to use this function to turn the seagulls into blobs based on their colour, get the coordinates of the blobs if we want to, or slice them our of the image for use later.

Let's apply some thresholding to the image, and eyeball the results:

{% highlight python %}

#Lets apply some thresholding to the image to see if we can isolate the Seagull shape!


lower_bound = 0

for i in range(10):
    
    _, thresh = cv2.threshold(gray, lower_bound, 255, cv2.THRESH_BINARY)
    
    lower_bound += 20
    
    cv2.imshow("Thresholded Image", thresh)
    
    cv2.waitKey(0)
    

{% endhighlight %}


This should hopefully give as an idea of the best threshold value to use, in order to maximizes the separation between the gulls' and the background

cv2.threshold allows us to apply a threshold to the image based on pixel (so from 0 - 255 in a grey image).


![ThresholdGradient](/assets/ThresholdGradient.jpg)



> 140 the Seagulls are clear visible.


