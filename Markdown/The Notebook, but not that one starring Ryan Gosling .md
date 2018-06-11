
![hello](Pictures/Adele-Hello.jpg)

# Topics
+ Data Presentation
+ Visualization
+ Optimization
+ Machine Learning
+ Misc.

# Data Visualization

## This is a Jupyter Notebook with RISE functionality

![jupyter](Pictures/jupyter.png)

### You might have noticed I'm running this in my browser. And yes, that does mean I can run it off a server (internal or external) and show off results to other people.
![browser](Pictures/browser.jpg)

### Jupyter is nice in that it allows for multiple programming languages. It's name is a portmanteau of Julia, Python and R. I use Python, but it normally looks like this:

![spyder](Pictures/Spyder.jpg)

Python has a lot of useful interactions with other programming languages that could be Hatch relevant

* SQL is the most obvious one. I can pull results from queries directly into Pandas dataframes for analysis
* It also supports JS, so I could in theory run the D3.js libraries directly out of this for really nice visuals
* It is possible to generate HTML with Python as well
* Python programs can be made into applications which can be run on servers (e.g. live reporting)

Python has several cool visualization tools

* Matplotlib
* Seaborn
* Bokeh


```python
#Here's a fairly basic matplotlib histogram. This is live code.
#This is just setup to get data in
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
id_df = pd.read_csv('C:/Users/Perry/.spyder-py3/indeed_hw_data.csv', header=0)
cols = id_df.columns.values.astype(str)
colct = np.arange(len(cols))
nancol = []
#missing revenue the same as zero    
id_df['revenue'] = id_df['revenue'].fillna(0)
id_age_ex = id_df.loc[id_df['age'] <= 0]
id_age_ex_sub = id_age_ex.loc[id_age_ex['age'] < 0]  
#limit to only zero age    
id_age_ex = id_age_ex.loc[id_age_ex['age'] == 0]
id_df = id_df.loc[id_df['age'] >= 0]
id_df = id_df.loc[id_df['revenue'] < 3500000000]
    
    
```


```python
#Now I can draw the figure 
num_bins = 100
gdt = id_df['revenue'].loc[id_df['revenue'] > 0]
gdt = gdt[gdt <= 150000000]
n, bins, patches = plt.hist(gdt, num_bins, density = 1, facecolor='blue', alpha=0.5)
plt.ticklabel_format(style='plain', axis='x', scilimits=(0,0))
plt.title("Revenue")
plt.xlabel("Value")
plt.ylabel("Frequency")
fig = plt.gcf()
fig.set_size_inches(18.5, 10.5)
```


![png](The%20Notebook%2C%20but%20not%20that%20one%20starring%20Ryan%20Gosling%20_files/The%20Notebook%2C%20but%20not%20that%20one%20starring%20Ryan%20Gosling%20_8_0.png)



```python
#We can quickly make this look nicer with Seaborn styles
import seaborn as sns

sns.set()
n, bins, patches = plt.hist(gdt, num_bins, density = 1, alpha=0.5)
plt.title("Revenue")
plt.xlabel("Value")
plt.ylabel("Frequency")
fig = plt.gcf()
fig.set_size_inches(18.5, 10.5)
```


![png](The%20Notebook%2C%20but%20not%20that%20one%20starring%20Ryan%20Gosling%20_files/The%20Notebook%2C%20but%20not%20that%20one%20starring%20Ryan%20Gosling%20_9_0.png)



```python
#It can be used to create some really nice looking visualizations
#This is an example from their site (again this is live code)
sns.set(style="white", rc={"axes.facecolor": (0, 0, 0, 0)})
rs = np.random.RandomState(1979)
x = rs.randn(500)
g = np.tile(list("ABCDEFGHIJ"), 50)
df = pd.DataFrame(dict(x=x, g=g))
m = df.g.map(ord)
df["x"] += m
pal = sns.cubehelix_palette(10, rot=-.25, light=.7)
g = sns.FacetGrid(df, row="g", hue="g", aspect=15, size=.5, palette=pal)
g.map(sns.kdeplot, "x", clip_on=False, shade=True, alpha=1, lw=1.5, bw=.2)
g.map(sns.kdeplot, "x", clip_on=False, color="w", lw=2, bw=.2)
g.map(plt.axhline, y=0, lw=2, clip_on=False)
def label(x, color, label):
    ax = plt.gca()
    ax.text(0, .2, label, fontweight="bold", color=color, 
            ha="left", va="center", transform=ax.transAxes)
g.map(label, "x")
g.fig.subplots_adjust(hspace=-.25)
g.set_titles("")
g.set(yticks=[])
g.despine(bottom=True, left=True)
```




    <seaborn.axisgrid.FacetGrid at 0x28f13269668>




![png](The%20Notebook%2C%20but%20not%20that%20one%20starring%20Ryan%20Gosling%20_files/The%20Notebook%2C%20but%20not%20that%20one%20starring%20Ryan%20Gosling%20_10_1.png)


# Bokeh is the one you'll be most interested in. It allows for the same sort of interactivity that D3.js does, but is in Python. I'd have to learn this, but it's a small lift since I do a lot of Python anyway


```python
#Here's an example from their site

from bokeh.io import output_notebook
from bokeh.plotting import figure, show

N = 4000

x = np.random.random(size=N) * 100
y = np.random.random(size=N) * 100
radii = np.random.random(size=N) * 1.5
colors = ["#%02x%02x%02x" %(int(r),int(g),150) for r, g in zip(np.floor(50+2*x), np.floor(30+2*y))]

output_notebook()

p = figure()
p.circle(x, y, radius = radii, fill_color=colors, fill_alpha=0.6, line_color=None)

show(p)
```



    <div class="bk-root">
        <a href="https://bokeh.pydata.org" target="_blank" class="bk-logo bk-logo-small bk-logo-notebook"></a>
        <span id="c7d33835-3e17-4402-a6b7-a22c7cee7a00">Loading BokehJS ...</span>
    </div>






<div class="bk-root">
    <div class="bk-plotdiv" id="5ee4afc5-9b7e-4639-a128-775e2e2abbca"></div>
</div>




# Optimization

### One of the more important skills I've picked up, Python allows for calculating the type of advanced curve optimizations I used to do in school with Matlab, but with access to better data structures and file IO


* purchase quantities
* bundles of items 
    - creating packages that maximize our profit 
    - getting the most dense carts for customers 
* labor allocations
    - CICs
    - buildroom
* account distribution

Smash.GG Fantasy

Given team budget and allowed to pick certain number of players. Very similar to auction leagues for fantasy football

This is my team for the Norcal Regionals Event. I came in second:

![NCR](Pictures/fantasy.jpg)

# I pick my teams by using linear optimization on player expected values

# I composite expected values from prior results with some manipulation based on my anecdotal knowledge of players

# There are multiple types of these optimizers beyond linear, but linear works well for this problem space and compiles quickly


```python
#This retroactively looked at the results from a major to calculate the best possible team
#I use the same code with almost zero changes to evaluate teams; it's immaterial to switch
import numpy as np
import pandas as pd
from pulp import *

df = pd.read_csv('players1.csv', header=None)
a = df[0]
b = df[4]
c = df[5]
tbl = np.array([a, b, c]).T

fantasy_budget = pulp.LpProblem('fantasy optimization', pulp.LpMaximize)

players = tbl[:,0]

x = pulp.LpVariable.dict('x_%s', players, lowBound =0, cat='Integer')

price = dict(zip(players, tbl[:,2]))

ev = dict(zip(players, tbl[:,1]))

fantasy_budget += sum([ (x[i]*price[i])*ev[i] for i in players])

fantasy_budget += sum([ (x[i]*price[i]) for i in players]) <= 1200

fantasy_budget += sum([ (x[i]) for i in players]) == 12
   
for player in players:
    fantasy_budget += x[player] <= price[player]    
  
for player in players:
    fantasy_budget += x[player] <= 1       

fantasy_budget.solve()

hld = []
for player in players:
    t = x[player]
    u = x[player].value()
    v = np.array([t, u])
    hld.append(v)

df = pd.DataFrame(hld, index=None, columns=None)  

print(df[(df[1] > 0)])
```

                       0    1
    0         x_SonicFox  1.0
    1              x_GO1  1.0
    2           x_Dogura  1.0
    3         x_NyChrisG  1.0
    4         x_Kazunoko  1.0
    30        x_TwiTchAy  1.0
    44  x_PSYKENonTWITCH  1.0
    46            x_Des!  1.0
    47            x_Nice  1.0
    61      x_Coolestred  1.0
    62   x_Coffee_prince  1.0
    63          x_Shogun  1.0
    

# This team would have performed >25% better than the single best human picked team even without bonus questions

# I of course got too clever for my own good and forced a player on my team and did not finish in prizes for this event


```python
from IPython.display import HTML

# Youtube
HTML('<iframe width="560" height="315" src="https://www.youtube.com/embed/vP65VVoUm6E" frameborder="0" allow="autoplay; encrypted-media" allowfullscreen></iframe>')
```




<iframe width="560" height="315" src="https://www.youtube.com/embed/vP65VVoUm6E" frameborder="0" allow="autoplay; encrypted-media" allowfullscreen></iframe>



# Machine Learning

# Recently a friend invited me to do some Optical Character Recognition work to help process match VODs into a repository

## The best players of the fighting game Guilty Gear live in Japan and play at the Mikado arcade

![mikado](Pictures/mikado.jpg)

## In the West we watch these VODs to learn how to play better. Unfortunately, what we get are unmarked videos of several hours in length that lack critical information

# Relevant Information

* Characters
* Match Start Time
* Players

### Luckily for me work has been done on all 3, but the current solution to the third is a paid web service (Google Vision API)

### Also fortunately, all revelant information is available on one screen, even though quality is quite low

![screen](Pictures/screen3.jpg)

# I was going to evaluate the relative performance between a local solution called PyTesseract to Google Vision API (which is world class performance)

### Tesseract is pretty good, and the new version uses LSTM trained sets, but it's pretty far behind Google theoretically as google is applying very sophisticated pre-processing to a cluster trained model based on 8+ layer convolutional neural networks (CNN)

### Process Image
![proc](Pictures/3656.png)


```python
from PIL import Image
import pytesseract
import argparse
import cv2
import os

pytesseract.pytesseract.tesseract_cmd = 'C:/Program Files (x86)/Tesseract-OCR/tesseract.exe'

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True,
	help="path to input image to be OCR'd")
ap.add_argument("-p", "--preprocess", type=str, default=None,
	help="type of preprocessing to be done")
args = vars(ap.parse_args())

#filename = os.path.join(skimage.data_dir, 'moon.png')

#filename = 'C:/Users/Perry/Pictures/screen3.jpg'
image = cv2.imread(args["image"])
image = cv2.resize(image, (1280,720), interpolation = cv2.INTER_AREA)
image2 = image[492:524, 46:409]
image = image[480:514, 825:1118]
imagec = []
imagec = imagec.append(image)
#imagec = imagec.append(image2)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


if args["preprocess"] == "thresh":
	gray = cv2.threshold(gray, 140, 255,
		cv2.THRESH_TOZERO)[1] #| cv2.THRESH_OTSU)[1]

# noise
elif args["preprocess"] == "blur":
	gray = cv2.medianBlur(gray, 3)
 

gray = util.invert(gray)

#gray = equalize_hist(gray)

filename = "{}.png".format(os.getpid())
cv2.imwrite(filename, gray)


text = pytesseract.image_to_string(Image.open(filename), lang='jpn+eng')#, config='--psm 10')
os.remove(filename)
print(text)
 
# show the output images
cv2.imshow("Image", image)
cv2.imshow("Output", gray)
cv2.waitKey(0)
```

    usage: ipykernel_launcher.py [-h] -i IMAGE [-p PREPROCESS]
    ipykernel_launcher.py: error: the following arguments are required: -i/--image
    


    An exception has occurred, use %tb to see the full traceback.
    

    SystemExit: 2
    


    C:\Users\Perry\AppData\Local\conda\conda\envs\Classification\lib\site-packages\IPython\core\interactiveshell.py:2918: UserWarning: To exit: use 'exit', 'quit', or Ctrl-D.
      warn("To exit: use 'exit', 'quit', or Ctrl-D.", stacklevel=1)
    

# This is a command line application, so I can't run it in this notebook, but this is basically what it outputs

![days](Pictures/days.jpg)

* Text output for use in creating upload script
* Unprocessed mask
* Processed mask

### How does this help Hatch?

### Hatch has a paper process in order entry, or often has things written down that need to go into the CRM/ERP in text form that currently have to be typed in

### Current commercial solutions are usually poor performers or require API knowledge. I can cover the latter and enable use of Google and MS' world class systems (they are actually quite reasonable, a year of scanning forms for Hatch would probably cost less than 10,000USD)

### It doesnt even matter if it's typed or handwritten; I have ML strategies for handling that because I can train datasets based on Hatch employee's handwriting which would significantly

### I can't match Google's precision, but I can beat Tesseract by using my own CNNs. In fact, I've trained my own using one of these:

![teneighty](Pictures/1080ti.jpg)

# ML more generally

## So I've spent the last year on ML

## Specifically, I've done signal classification of EEG signals for the purposes of detecting seizures in real time

## Considering how SSG works, ML may be the only way to approximate the correct order of SSG games. 

## Because I built SSG progress into a SQL table (i.e. a 2-dimensional matrix), I can iteratively use each column to estimate the score kids would get on another game

## Basically it's ripping off Netflix's recommendation system. It really doesn't matter what specific engine you use either as it's a pretty low-complexity problem, so you can use XGBoost or another tabular engine that runs quickly on CPUs to avoid additional hardware cost

## For what it's worth, the same thing can be applied to recommending items to customers based on the length of time since they've purchased. It would be stronger if Hatch had more proprietary stuff, but it should probably work out anyway

## TheDialer/Skynet could be improved with ML by updating estimators based on call success 

## There could theoretically be a Dragon v3

## Since my work has mainly been in classification, I would want to get ML based predictions of account Expected Values

# Miscellaneous


### Image Classification

### Signal Processing specifically

### Obviously I still have the skills to do reporting and tackle the same sort of SQL problems I did before (e.g. renewals revenue reporting)

### Questions?
