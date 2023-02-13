#!/usr/bin/env python
# coding: utf-8

# # Deep learning Image Recognizer
# ## Is it a Cat or a Dog?

# In[1]:


from fastai.vision.all import *
path = untar_data(URLs.PETS)/'images'

def is_cat(x): return x[0].isupper()
dls = ImageDataLoaders.from_name_func(
    path, get_image_files(path), valid_pct=0.2, seed=42,
    label_func=is_cat, item_tfms=Resize(224))

learn = vision_learner(dls, resnet34, metrics=error_rate)
learn.fine_tune(1)


# #### Imported library with computer vision models
# #### The downloaded dataset contains images of dogs and cats and it is used to train the model. A random validation set of 20% of the data is held out and the remaining 80% is used to train the model. The accuracy of the model is measured by the validation set, which is used to make sure the model is not memorizing the data instead of finding generalizable patterns in it.
# #### A class such as ImageDataLoader, which can be used to load and process deep learning datasets for image recognition tasks. This class takes in information such as the type of data, how to get the labels from the dataset, and what transforms to apply. For example, we can tell fastai to resize all images to 224 pixels, which is the standard size for most historical models. Additionally, item_tfms and batch_tfms can be applied to each image, or batches of images, to help improve the accuracy of the model.
# #### object learn creates a deep learning model with a convolutional neural network architecture to train on an image dataset, and uses an accuracy metric to measure how well it does.
# #### learn.fine_tune method adapt a pre-trained model to a new dataset. It involves adjusting the parameters of the model to better fit the data, without having to start from scratch. This allows us to take advantage of the pretrained model's capabilities, while still having the flexibility to adjust it to fit our specific needs.
# 

# In[7]:


from fastbook import *
img = PILImage.create(image_cat())
img.to_thumb(192)


# In[9]:


#hide_output
uploader = widgets.FileUpload()
uploader


# In[10]:


#hide
uploader = SimpleNamespace(data = ['images/cat.jpeg'])


# In[11]:


img = PILImage.create(uploader.data[0])
is_cat,_,probs = learn.predict(img)
print(f"Is this a cat?: {is_cat}.")
print(f"Probability it's a cat: {probs[1].item():.6f}")


# In[12]:


uploader = SimpleNamespace(data = ['images/dog.jpeg'])


# In[13]:


img = PILImage.create(uploader.data[0])
is_cat,_,probs = learn.predict(img)
print(f"Is this a cat?: {is_cat}.")
print(f"Probability it's a cat: {probs[1].item():.6f}")


# # Segmentation model

# In[14]:



path = untar_data(URLs.CAMVID_TINY)
dls = SegmentationDataLoaders.from_label_func(
    path, bs=8, fnames = get_image_files(path/"images"),
    label_func = lambda o: path/'labels'/f'{o.stem}_P{o.suffix}',
    codes = np.loadtxt(path/'codes.txt', dtype=str)
)

learn = unet_learner(dls, resnet34)
learn.fine_tune(8)
     


# In[16]:


learn.show_results(max_n=6, figsize=(10,12))


# #### A segmentation model is a type of machine learning model that is used to identify the content of each pixel in an image. For example, it can be used to recognize objects like cars and trees in a picture. To train such a model, we can use a dataset such as the Camvid dataset. After training, the model can accurately classify each pixel in an image, providing a color-coded representation of the objects in the image

# # Build models from csv

# In[28]:


from fastai.tabular.all import *
path = untar_data(URLs.ADULT_SAMPLE)

dls = TabularDataLoaders.from_csv(path/'adult.csv', path=path, y_names="salary",
    cat_names = ['workclass', 'education', 'marital-status', 'occupation',
                 'relationship', 'race'],
    cont_names = ['age', 'fnlwgt', 'education-num'],
    procs = [Categorify, FillMissing, Normalize])

learn = tabular_learner(dls, metrics=accuracy)
     


# #### To train a model to predict whether a person is a high-income earner based on their socioeconomic background, we need to define which columns of data contain categorical values (like occupation) and which contain continuous values (like age). We then use a method called fit_one_cycle to train the model from scratch, without relying on a pre-trained model.

# In[31]:


dls.show_batch()


# In[30]:


learn.fit_one_cycle(3)


# #### This model is able to accurately predict whether or not an individual has an annual income greater than $50k based on their demographic data. It takes around 10 seconds to train, and is over 80% accurate.

# # Movie Recommendation System
# ## Collaborative filtering

# In[33]:


from fastai.collab import *
path = untar_data(URLs.ML_SAMPLE)
dls = CollabDataLoaders.from_csv(path/'ratings.csv')
learn = collab_learner(dls, y_range=(0.5,5.5))
learn.fine_tune(10)


# #### Recommendation systems use data to suggest products or movies that people might like. To train a model that will predict what movies people might like, based on their viewing habits, we can use the MovieLens dataset. This dataset contains information about users, ratings, and genres of movies, which can be used to build a model that predicts what movies people are likely to enjoy.
# ####  This model can be used to predict movie ratings on a scale of 0.5 to 5.0. By using the y_range parameter, using fine_tune feature to make predictions with an average error of around 0.6.

# In[34]:


dls.show_batch()


# In[35]:


learn.show_results()


# #### show_results is a way to view examples of user and movie IDs, actual ratings, and predictions. This shows a better understanding of how a predictive model is performing.

# In[36]:


learn.fit_one_cycle(3)


# In[37]:


learn.show_results()


# In[ ]:




