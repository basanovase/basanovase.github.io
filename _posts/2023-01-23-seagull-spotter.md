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

These are definitely questions that can't go unanswered. Join me on a pointless journey into computer vision, machine learning and using the latest technologies to answer questions that definitely don't need to be answered. 

<B>THE PROBLEM</B>:

To start to address some of these burning questions, we first need a consistent view of the birds and their environment. If you're like me and have and have too much tech lying around, one of the ESP32 microcontrollers seems like a great fit for the task.

![ESP32](assets/esp32.jpeg)

<b>EXPLORATIONS USING OPENCV</b>:

Rather that racing straight to using a supervised learning solution - I thought it would be fun to explore how far we can get with computer vision.


An example image:

![EXAMPLE](/assets/OriginalImage.jpg)

There's quite a lot going on here. We'll need to do some preprocessing to get things in a suitable state for some OpenCV magic.

Given Seagulls are black and white, I think we can convert the image to greyscale. This will reduce the amount of data we need to process and some noise in the image. Helpfully, Seagulls are also black and white. The list of positive things I can say about seagulls grows!

1.
{% highlight python %}
import cv2

def convert_to_grayscale(image):
        grey_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return grey_image

{% endhighlight %}

2.  
Looking into how we can better recognize Seagulls using the OpenCV library requires giving some thought about the shape of seagulls. Given I don't really want to spend too long thinking about seagulls, I decided to go with the most obvious thing based on the data to date.

Seagulls are usually wider than they are tall:

![SEAGULLPROOF](/assets/SeagullProof.jpg)
You can enable MathJax by setting `mathjax: true` on a page or globally in the `_config.yml`. Some examples:

[Euler's formula](https://en.wikipedia.org/wiki/Euler%27s_formula) relates the  complex exponential function to the trigonometric functions.

$$ e^{i\theta}=\cos(\theta)+i\sin(\theta) $$

The [Euler-Lagrange](https://en.wikipedia.org/wiki/Lagrangian_mechanics) differential equation is the fundamental equation of calculus of variations.

$$ \frac{\mathrm{d}}{\mathrm{d}t} \left ( \frac{\partial L}{\partial \dot{q}} \right ) = \frac{\partial L}{\partial q} $$

The [Schrödinger equation](https://en.wikipedia.org/wiki/Schr%C3%B6dinger_equation) describes how the quantum state of a quantum system changes with time.

$$ i\hbar\frac{\partial}{\partial t} \Psi(\mathbf{r},t) = \left [ \frac{-\hbar^2}{2\mu}\nabla^2 + V(\mathbf{r},t)\right ] \Psi(\mathbf{r},t) $$

## Code

Embed code by putting `{{ "{% highlight language " }}%}` `{{ "{% endhighlight " }}%}` blocks around it. Adding the parameter `linenos` will show source lines besides the code.

{% highlight c %}

static void asyncEnabled(Dict* args, void* vAdmin, String* txid, struct Allocator* requestAlloc)
{
    struct Admin* admin = Identity_check((struct Admin*) vAdmin);
    int64_t enabled = admin->asyncEnabled;
    Dict d = Dict_CONST(String_CONST("asyncEnabled"), Int_OBJ(enabled), NULL);
    Admin_sendMessage(&d, txid, admin);
}

{% endhighlight %}

## Gists

With the `jekyll-gist` plugin, which is preinstalled on Github Pages, you can embed gists simply by using the `gist` command:

<script src="https://gist.github.com/5555251.js?file=gist.md"></script>

## Images

Upload an image to the *assets* folder and embed it with `![title](/assets/name.jpg))`. Keep in mind that the path needs to be adjusted if Jekyll is run inside a subfolder.

A wrapper `div` with the class `large` can be used to increase the width of an image or iframe.

![Flower](https://user-images.githubusercontent.com/4943215/55412447-bcdb6c80-5567-11e9-8d12-b1e35fd5e50c.jpg)

[Flower](https://unsplash.com/photos/iGrsa9rL11o) by Tj Holowaychuk

## Embedded content

You can also embed a lot of stuff, for example from YouTube, using the `embed.html` include.

{% include embed.html url="https://www.youtube.com/embed/_C0A5zX-iqM" %}
