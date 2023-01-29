---
title:  "Watching birds on the beach"
mathjax: true
layout: post
categories: media
---

![Swiss Alps](https://user-images.githubusercontent.com/4943215/55412536-edbba180-5567-11e9-9c70-6d33bca3f8ed.jpg)




There's a lot of birds on the beach I live in front of.  Naturally, questions arise about the seagulls:

How many birds are there? Are they always on the beach? Are they having a good time? 

These are definitely questions that can't go unanswered. Join me on a pointless journey into computer vision, machine learning and using the latest technologies to answer questions that definitely don't need to be answered. Why? Why not!

## THE PROBLEM:

To start to address some of these burning questions, we first need a consistent view of the birds and their environment. If you're like me and have and have far too much tech lying around, one of the ESP32 microcontrollers seems like a great fit for the task. We can then add sensors to collect data to our hearts content.

![ESP32](/assets/esp32.jpg)

## EXPLORATIONS USING OPENCV:

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



Greater than 140 the Seagulls are clear visible.


Nice! Let's apply our code to date:



{% highlight python %}

#Lets apply some thresholding to the image to see if we can isolate the Seagull shape!


lower_bound = 0

#Read in the image
image = cv2.imread("/Users/flynnmclean/Downloads/20230120_181747.jpg")
#Convert it to grey
grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

_, thresh = cv2.threshold(grey, 230, 255, cv2.THRESH_BINARY)

seagull_count = 0

nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(thresh, connectivity=8)

for i in range(1, nb_components):
    size = stats[i, cv2.CC_STAT_AREA]
    #if size < 370:
        #continue
    x, y, w, h, = stats[i, cv2.CC_STAT_LEFT], stats[i, cv2.CC_STAT_TOP], stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT]
    
    #Create an aspect ratio for later use
    aspect_ratio = float(w) / h
    #SEAGULLS ARE USUALL WIDER THAN THE ARE TALLw
   
        
        #Make the crop bigger to capture the whole gull, nobody likes a partial gulls, am I right?

    seagull_count += 1
        #Grab the crop so the rectangle HASNT been drawn
 
    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)  # Draw a rectangle around the seagull
 
cv2.imshow('connectedComponentsWithStats',image)

cv2.imwrite('/Users/flynnmclean/Documents/Projects/BirdFinder/connectedComponentsWithStats.jpg', image)
print(f'Seagull count: {seagull_count}')
cv2.waitKey(0)
    

{% endhighlight %}

<i>Seagull count: 16326</i> - wowza, that's an outrageous amount of seagulls. Hmm.

Eyeballing the image I can see we're actually detected most of the seagulls which is great, but we appear to be also detecting all the smaller patches of white:

![SEAGULLPROOF](/assets/connectedComponentsWithStats.jpg)

We will need to apply some further OpenCV magic to clean up these smaller shapes. Remember our hilarious seagull juxtaposition from earlier? We'll use that theory to apply some logic based on the aspect ratio of the connectedComponent:


{% highlight python %}

image = cv2.imread("/Users/flynnmclean/Downloads/20230120_181747.jpg")
#Convert it to grey
grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

_, thresh = cv2.threshold(grey, 230, 255, cv2.THRESH_BINARY)

seagull_count = 0

nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(thresh, connectivity=8)

for i in range(1, nb_components):
    size = stats[i, cv2.CC_STAT_AREA]
    #if size < 370:
        #continue
    x, y, w, h, = stats[i, cv2.CC_STAT_LEFT], stats[i, cv2.CC_STAT_TOP], stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT]
    aspect_ratio = float(w) / h
    #SEAGULLS ARE USUALL WIDER THAN THE ARE TALLw
    if aspect_ratio > 0.8 and aspect_ratio < 1.2:
        
        #Make the crop bigger to capture the whole gull, nobody likes a partial gulls, am I right?

        seagull_count += 1
      
 
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)  # Draw a rectangle around the seagull
 
cv2.imshow('connectedComponentsWithStats',image)

cv2.imwrite('/Users/flynnmclean/Documents/Projects/BirdFinder/connectedComponentsWithStats.jpg', image)
print(f'Seagull count: {seagull_count}')
cv2.waitKey(0)

{% endhighlight %}

<i>Seagull count: 5671</i>

Certainly a more reasonable amount of Seagulls

I think we can do better though - we can also apply some logic to only plot the larger connectedComponents. A quick google reveals there's also a couple of easy OpenCV filters we can apply to the image.

let's also clean it up into a nicer class structure for the Pythonistas:

{% highlight python %}

class SeagullDetector:
    def __init__(self, image):
        self.image = image
        self.seagull_count = 0
        self.seagull_crops = []

    def convert_to_grayscale(self):
        self.gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)

    def threshold_image(self):
        _, self.thresh = cv2.threshold(self.gray, 230, 255, cv2.THRESH_BINARY)

    def perform_size_filtering(self):
        nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(self.thresh, connectivity=8)
        for i in range(1, nb_components):
            size = stats[i, cv2.CC_STAT_AREA]
            #if size < 370:
                #continue
            x, y, w, h, = stats[i, cv2.CC_STAT_LEFT], stats[i, cv2.CC_STAT_TOP], stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT]
            aspect_ratio = float(w) / h
            #SEAGULLS ARE USUALL WIDER THAN THE ARE TALLw
            if aspect_ratio > 0.8 and aspect_ratio < 1.2:
                
                #Make the crop bigger to capture the whole gull, nobody likes a partial gulls, am I right? Covered later on
                x, y, w, h = increase_crop_size(x, y, w, h, self.image.shape[1],  self.image.shape[0])
                self.seagull_count += 1
                #Grab the crop so the rectangle HASNT been drawn
                #Covered later on!
                seagull_crop = self.image[y:y+h, x:x+w]
                cv2.rectangle(self.image, (x, y), (x + w, y + h), (0, 255, 0), 2)  # Draw a rectangle around the seagull
                self.seagull_crops.append(seagull_crop)

    def detect_seagulls(self):
        self.convert_to_grayscale()
        self.threshold_image()
        self.perform_size_filtering()
        print(self.seagull_count)
        return self.seagull_count, self.image

{% endhighlight %}

Google and stackoverflow have advised there are a couple of operations that should "reduce the size of smaller" components


{% highlight python %}


cv2.erode()
cv2.dilate()

{% endhighlight %}



I'll apply these as a function to the class so I can toggle them on and off and gauge the impact. I'm also going to add a list of the crops so I can use them later if I want to do any classification tasks.

I'm also going to add a function to increase the size of the snip, so that the snipped image extends past the edges of the whole detected object. I'll also add a function to centre the snip more. We'll also add in the size filtering we discussed.

{% highlight python %}


import cv2
import numpy as np
import os



def increase_crop_size(x, y, w, h, image_shape_x, image_shape_y):
    
            #Increase the crop size
            x = int(x - 1.2 * w)
            y = int(y - 1.2 * h)
            w = int(2.2 * w)
            h = int(2.2 * h)
            
            # Ensure that the crop remains within the bounds of the image
            x = max(x, 0)
            y = max(y, 0)
            w = min(w, image_shape_x - x)
            h = min(h, image_shape_y - y)
            
            return x, y, w, h


def centre_image(x, y, w, h, image_shape_x, image_shape_y):
    
    #Use the centroid of each seagull to shift the axis of the crop. 
    x_centroid = int(x + (w/2))
    y_centroid = int(y + (h/2))
    
    #Shift the crop so that the seagull is centred
    x = int(x_centroid - (w/4))
    y = int(y_centroid - (h/4))
    
    #Ensure that the crop remains within the bounds of the image
    x = max(x, 0)
    y = max(y, 0)
    w = min(w, image_shape_x - x)
    h = min(h, image_shape_y - y)
    
    return x, y, w, h




class SeagullDetector:
    
    def __init__(self, image):
        self.image = image
        self.original_image = image
        self.seagull_count = 0
        self.seagull_crops = []

    def convert_to_grayscale(self):
        self.gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)

    def threshold_image(self):
        _, self.thresh = cv2.threshold(self.gray, 170, 255, cv2.THRESH_BINARY)

    def perform_morphological_operations(self):
        kernel = np.ones((5,5),np.uint8)
        erosion = cv2.erode(self.thresh, kernel, iterations = 1)
        dilation = cv2.dilate(erosion, kernel, iterations = 1)
        self.thresh = dilation

    def perform_size_filtering(self):
        nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(self.thresh, connectivity=8)
        for i in range(1, nb_components):
            size = stats[i, cv2.CC_STAT_AREA]
     
            if size < self.image.shape[1]*0.28:
                continue
            x, y, w, h, = stats[i, cv2.CC_STAT_LEFT], stats[i, cv2.CC_STAT_TOP], stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT]
            aspect_ratio = float(w) / h
            #SEAGULLS ARE USUALL WIDER THAN THE ARE TALLw
            if aspect_ratio > 0.8 and aspect_ratio < 1.2:
                
                #Make the crop bigger to capture the whole gull, nobody likes a partial gulls, am I right?
                x, y, w, h = increase_crop_size(x, y, w, h, self.image.shape[1],  self.image.shape[0])
                
                x, y, w, h = centre_image(x, y, w, h, self.image.shape[1], self.image.shape[0])
                self.image_copy = self.image.copy()
                self.seagull_count += 1
                #Grab the crop so the rectangle HASNT been drawn
                seagull_crop = self.image[y:y+h, x:x+w]
                
                cv2.rectangle(self.image_copy, (x, y), (x + w, y + h), (0, 255, 0), 2)  # Draw a rectangle around the seagull
         
                self.seagull_crops.append(seagull_crop)

    def detect_seagulls(self):
        self.convert_to_grayscale()
        self.threshold_image()
        self.perform_morphological_operations()
        self.perform_size_filtering()
        print(self.seagull_count)
        return self.seagull_count, self.image





if __name__ == "__main__":

    img = cv2.imread("/Users/flynnmclean/Downloads/20230120_181747.jpg")
    
    isd = SeagullDetector(img)
    
    test = isd.detect_seagulls()

    cv2.imshow('SEAGULLIMAGE', isd.image)
    
    cv2.waitKey(0)
    
    for item in isd.seagull_crops:
        
        cv2.imshow('SEAGULL CROP', item)
        
        cv2.waitKey(0)

{% endhighlight %}

<i>Seagull count: 85</i>

Nice! With those changes we seem to be at a <i>reasonable</i> number. We would need further labelled data to check accuracy, but I think the result based on the sample images is OK is pretty darn good for just computer vision alone! 

Here's the world's most annoying montage:


![LAMEMONTAGE](/assets/LameMontage.jpg)

Let's also add a montage function, because at this point, why not, and a feature the use the webcam prior to shifting over to the ESP32 device for testing. 

{% highlight python %}

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 23 21:29:09 2023

@author: flynnmclean
"""



import cv2
import numpy as np
import os
from PIL import Image


def stream_from_webcam():
    
    import cv2

    # Create an instance of the SeagullDetector class
   
    
    # Open a connection to the webcam
    cap = cv2.VideoCapture(0)
    
    # Create a window to display the output
    cv2.namedWindow("Seagull Detection", cv2.WINDOW_NORMAL)
    
    while True:
        # Read a frame from the webcam
        ret, frame = cap.read()
        
        isd = SeagullDetector(frame)
        # Detect seagulls in the frame
        seagull_count, output_image = isd.detect_seagulls()
        
        # Show the output image
        #cv2.imshow("Seagull Detection", output_image)
        
        for x, y, w, h in isd.seagull_bboxs:
            
            cv2.rectangle(output_image, (x, y), (x + w, y + h), (0, 255, 0), 2)
    
        # Show the output image
       
        cv2.putText(output_image, "Gull count: {}".format(isd.seagull_count), (10, output_image.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    
        cv2.imshow("Seagull Detection", output_image)
        
        # Exit the loop when the 'q' key is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        
        
    
    # Release the webcam and close the window
    cap.release()
    cv2.destroyAllWindows()

def create_montage(image_list):
    images = [Image.fromarray(image) for image in image_list]
    widths, heights = zip(*(i.size for i in images))
    total_width = sum(widths)
    max_height = max(heights)

    new_im = Image.new('RGB', (total_width, max_height))

    x_offset = 0
    for im in images:
        new_im.paste(im, (x_offset,0))
        x_offset += im.size[0]

    return new_im


def increase_crop_size(x, y, w, h, image_shape_x, image_shape_y):
    
            #Increase the crop size
            x = int(x - 1.2 * w)
            y = int(y - 1.2 * h)
            w = int(2.2 * w)
            h = int(2.2 * h)
            
            # Ensure that the crop remains within the bounds of the image
            x = max(x, 0)
            y = max(y, 0)
            w = min(w, image_shape_x - x)
            h = min(h, image_shape_y - y)
            
            return x, y, w, h


def centre_image(x, y, w, h, image_shape_x, image_shape_y):
    
    #Use the centroid of each seagull to shift the axis of the crop. 
    x_centroid = int(x + (w/2))
    y_centroid = int(y + (h/2))
    
    #Shift the crop so that the seagull is centred
    x = int(x_centroid - (w/5))
    y = int(y_centroid - (h/5))
    
    #Ensure that the crop remains within the bounds of the image
    x = max(x, 0)
    y = max(y, 0)
    w = min(w, image_shape_x - x)
    h = min(h, image_shape_y - y)
    
    return x, y, w, h




class SeagullDetector:
    
    def __init__(self, image):
        self.image = image
        self.original_image = image
        self.seagull_count = 0
        self.seagull_crops = []
        self.seagull_bboxs = []
        
    def convert_to_grayscale(self):
        self.gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)

    def threshold_image(self):
        _, self.thresh = cv2.threshold(self.gray, 170, 255, cv2.THRESH_BINARY)

    def perform_morphological_operations(self):
        kernel = np.ones((5,5),np.uint8)
        erosion = cv2.erode(self.thresh, kernel, iterations = 1)
        dilation = cv2.dilate(erosion, kernel, iterations = 1)
        self.thresh = dilation

    def perform_size_filtering(self):
        nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(self.thresh, connectivity=8)
        for i in range(1, nb_components):
            size = stats[i, cv2.CC_STAT_AREA]
     
            if size < self.image.shape[1]*0.05:
                continue
            x, y, w, h, = stats[i, cv2.CC_STAT_LEFT], stats[i, cv2.CC_STAT_TOP], stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT]
            aspect_ratio = float(w) / h
            #SEAGULLS ARE USUALL WIDER THAN THE ARE TALLw
            if aspect_ratio > 0.8 and aspect_ratio < 1.2:
                
                #Make the crop bigger to capture the whole gull, nobody likes a partial gulls, am I right?
                x, y, w, h = increase_crop_size(x, y, w, h, self.image.shape[1],  self.image.shape[0])
                
                x, y, w, h = centre_image(x, y, w, h, self.image.shape[1], self.image.shape[0])
                
                self.seagull_bboxs.append([x, y, w, h])
                self.image_copy = self.image.copy()
                self.seagull_count += 1
                #Grab the crop so the rectangle HASNT been drawn
                seagull_crop = self.image[y:y+h, x:x+w]
                
                cv2.rectangle(self.image_copy, (x, y), (x + w, y + h), (0, 255, 0), 2)  # Draw a rectangle around the seagull
         
                self.seagull_crops.append(seagull_crop)

    def detect_seagulls(self):
        self.convert_to_grayscale()
        self.threshold_image()
        self.perform_morphological_operations()
        self.perform_size_filtering()
        print(self.seagull_count)
        return self.seagull_count, self.image





if __name__ == "__main__":

    stream_from_webcam()

        

{% endhighlight %}


## EXPLORATIONS USING CNN

A convolutional neural network (CNN) is a good fit for this task. The general idea is to train a CNN on a large dataset of images labeled with seagulls, or no seagulls. Given the weekend is running out I'm just going to use a pre-trained model like YOLO.


This was pretty interesting - I ended up initially trying the OpenCv implementation of YOLO but found this was clunky and not configurable enough.


What I ended up trying was the javascript implementation of Tensorflow (A good opportunity to improve my JS skills!) - this was really cool as you can literally just add the script tagg to an .html file with leveraging some basic built-in image recognition, and bam you've got object recognition happening in your browser. The library is really easy to use and intuitive, and a similar concept of just passing in an image and gett bounding boxes that should be plotted. I also like this as I could use an old broken phone with zooming capability to narrow down the monitoring area.


Very cool! - full example:


{% include tfjs.html %}





